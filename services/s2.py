from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
import rasterio
from pyproj import Transformer
from config import settings
from services.progress import reset as progress_reset, set_progress


# ---------------- helpers: bands / rgb quicklook ----------------

def _find_band(jp2_dir: Path, suffix: str) -> Path:
    for p in jp2_dir.glob(f"*_{suffix}_10m.jp2"):
        return p
    raise FileNotFoundError(f"{suffix} not found in {jp2_dir}")

def _build_rgb_geotiff() -> None:
    arrays = []
    profile = None
    for key in ("R", "G", "B"):
        path = _find_band(settings.S2_JP2_DIR, settings.S2_BANDS[key])
        with rasterio.open(path) as src:
            arrays.append(src.read(1))
            if profile is None:
                profile = src.profile.copy()
    rgb = np.stack(arrays)
    profile.update(count=3)
    with rasterio.open(settings.S2_RGB_TIF, "w", **profile) as dst:
        dst.write(rgb)

def _linear_stretch(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, (2, 98))
    if hi <= lo:
        lo, hi = float(a.min()), float(max(a.max(), 1))
    a = np.clip((a - lo) / (hi - lo + 1e-9), 0, 1)
    return (a * 255).astype(np.uint8)

def _save_quicklook_from_tif(max_dim=4096):
    with rasterio.open(settings.S2_RGB_TIF) as src:
        r, g, b = src.read()
    img = np.dstack([_linear_stretch(r), _linear_stretch(g), _linear_stretch(b)])
    im = Image.fromarray(img)
    W, H = im.size
    if max(W, H) > max_dim:
        scale = max_dim / float(max(W, H))
        im = im.resize((int(W*scale), int(H*scale)), Image.BILINEAR)
    im.save(settings.BACKDROP_IMAGE)

def ensure_backdrop():
    if settings.BACKDROP_IMAGE.exists():
        return
    try:
        if settings.S2_JP2_DIR and settings.S2_JP2_DIR.exists():
            if not settings.S2_RGB_TIF.exists():
                _build_rgb_geotiff()
            _save_quicklook_from_tif()
    except Exception as e:
        print('[WARN] Quicklook build failed:', e)
    if not settings.BACKDROP_IMAGE.exists():
        Image.new('RGB', (2048, 2048), (30, 30, 30)).save(settings.BACKDROP_IMAGE)

def backdrop_meta():
    im = Image.open(settings.BACKDROP_IMAGE).convert('RGB')
    return im.size  # (w, h)

def s2_bounds_wgs84():
    if not settings.S2_RGB_TIF.exists():
        return None
    with rasterio.open(settings.S2_RGB_TIF) as src:
        bounds = src.bounds
        crs = src.crs
    t = Transformer.from_crs(crs, 4326, always_xy=True)
    lon_min, lat_min = t.transform(bounds.left, bounds.bottom)
    lon_max, lat_max = t.transform(bounds.right, bounds.top)
    return {
        'lon_min': float(lon_min), 'lat_min': float(lat_min),
        'lon_max': float(lon_max), 'lat_max': float(lat_max)
    }

# ---------------- NDVI (L2A: offset/scale) ----------------

def _read_band_l2a(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        dn = src.read(1).astype(np.float32)
        nod = src.nodata
    bad0, bad1 = settings.BAD_DN_VALUES
    bad = (dn == bad0) | (dn == bad1)
    if nod is not None:
        bad |= (dn == float(nod))
    dn[bad] = np.nan
    # reflectance = (DN + BOA_ADD_OFFSET)/BOA_QUANT  → (DN - 1000)/10000
    refl = (dn + settings.BOA_ADD_OFFSET) / settings.BOA_QUANT
    return np.clip(refl, -0.2, 1.2)

def _jp2(glob_pat: str) -> Path:
    for p in settings.S2_JP2_DIR.glob(glob_pat):
        return p
    raise FileNotFoundError(f"JP2 not found: {glob_pat} in {settings.S2_JP2_DIR}")

def _compute_ndvi():
    if not settings.S2_JP2_DIR or not settings.S2_JP2_DIR.exists():
        raise RuntimeError("S2_JP2_DIR is not set or doesn't exist.")
    p_red = _jp2("*_B04_10m.jp2")
    p_nir = _jp2("*_B08_10m.jp2")
    print("[NDVI] RED:", p_red)
    print("[NDVI] NIR:", p_nir)
    red = _read_band_l2a(p_red)
    nir = _read_band_l2a(p_nir)
    if red.shape != nir.shape:
        raise RuntimeError(f"Shape mismatch: {red.shape} vs {nir.shape}")
    den = nir + red
    ndvi = (nir - red) / np.where(den == 0, np.nan, den)
    ndvi = np.clip(ndvi, -1.0, 1.0).astype(np.float32)
    h, w = ndvi.shape
    return ndvi, w, h

# ---------------- KMeans (MiniBatch + downscale + chunked predict) ----------------

def _kmeans_mask_from_rgb(
    arr_rgb: np.ndarray,
    n_clusters: int = 2,
    fit_max_side: int = 2048,
    batch_size: int = 16384,
    max_iter: int = 100,
    random_state: int = 0,
    predict_chunk_px: int = 10_000_000,
    ndvi_for_label: np.ndarray | None = None,
    progress_cb = None,                 # ⬅️ کال‌بک پیشرفت (phase:str, percent:float, detail:str)
    progress_range: tuple[float,float] = (10.0, 85.0)  # ⬅️ درصدها را در این بازه نگاشت می‌کنیم
) -> np.ndarray:
    """
    MiniBatchKMeans روی RGB با:
      - fit روی نسخه کوچک‌شده برای کاهش هزینه
      - predict به‌صورت chunked روی کل تصویر
      - انتخاب خوشهٔ گیاهی با NDVI (اگر باشد) یا روشنایی مراکز خوشه

    progress_cb:  تابعی مثل  progress_cb(phase, percent, detail='')
    progress_range: بازهٔ درصدی که این تابع مصرف می‌کند (باقی درصدها را caller تنظیم می‌کند)
    """
    try:
        from sklearn.cluster import MiniBatchKMeans
    except ImportError as e:
        if progress_cb:
            progress_cb("kmeans_error", 100.0, "Scikit-learn نصب نیست")
        raise

    H, W = arr_rgb.shape[:2]
    p0, p1 = progress_range
    p0 = float(p0); p1 = float(p1)
    if p1 < p0: p0, p1 = p1, p0

    def _emit(phase: str, frac: float, detail: str = ""):
        # frac در [0..1] → نگاشت به [p0..p1]
        frac = max(0.0, min(1.0, float(frac)))
        pct = p0 + (p1 - p0) * frac
        if progress_cb:
            progress_cb(phase, pct, detail)

    # 1) Fit روی نسخه کوچک‌شده
    _emit("kmeans_prepare", 0.00, "آماده‌سازی داده برای fit")
    img = Image.fromarray(arr_rgb.astype(np.uint8), mode='RGB')
    if max(H, W) > fit_max_side:
        scale = fit_max_side / float(max(H, W))
        fit_size = (int(W*scale), int(H*scale))
        img_fit = img.resize(fit_size, Image.BILINEAR)
    else:
        img_fit = img

    X_fit = np.asarray(img_fit, dtype=np.float32).reshape(-1, 3) / 255.0
    _emit("kmeans_fit", 0.10, f"تعداد نمونه fit: {X_fit.shape[0]:,}")

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        n_init='auto',
        max_iter=max_iter,
        random_state=random_state,
        verbose=0
    )
    km.fit(X_fit)
    _emit("kmeans_fit", 0.30, "fit تمام شد")

    # 2) Predict روی کل تصویر (chunked)
    X_full = arr_rgb.reshape(-1, 3).astype(np.float32) / 255.0
    y_full = np.empty(X_full.shape[0], dtype=np.int32)
    N = X_full.shape[0]
    _emit("kmeans_predict", 0.35, f"شروع predict روی {N:,} پیکسل")

    start = 0
    # رِنج 35%→85% را با پیشرفت چانک‌ها پر می‌کنیم (داخل همین تابع نگاشت می‌شود)
    while start < N:
        end = min(N, start + predict_chunk_px)
        y_full[start:end] = km.predict(X_full[start:end])
        start = end
        frac = start / N  # [0..1]
        # نگاشت داخلی: 0.35 → 0.85
        sub = 0.35 + 0.50 * frac
        _emit("kmeans_predict", sub, f"پیش‌بینی: {start:,}/{N:,}")

    y_full = y_full.reshape(H, W)
    _emit("kmeans_predict", 0.85, "predict تمام شد")

    # 3) تعیین خوشه‌ی گیاهی
    if ndvi_for_label is not None:
        _emit("kmeans_label", 0.90, "برچسب‌گذاری خوشه‌ها با NDVI")
        rng = np.random.default_rng(0)
        take = min(H*W//100, 200_000)
        idx = rng.choice(H*W, size=take, replace=False)
        cl = y_full.reshape(-1)[idx]
        nv = ndvi_for_label.reshape(-1)[idx]
        means = []
        for k in range(n_clusters):
            sel = (cl == k)
            m = float(np.nanmean(nv[sel])) if np.any(sel) else -1e9
            means.append(m)
        veg_cluster = int(np.argmax(means))
    else:
        _emit("kmeans_label", 0.90, "برچسب‌گذاری با روشنایی مرکز خوشه")
        centers = km.cluster_centers_  # [0..1]
        brightness = centers.mean(axis=1)
        veg_cluster = int(np.argmin(brightness))  # فرض: تیره‌تر=گیاهی

    mask = (y_full == veg_cluster).astype(np.uint8)
    _emit("kmeans_done", 1.00, "KMeans به پایان رسید")
    return mask

# ---------------- main entry: prelabel ----------------

def prelabel(method: str, **kwargs):
    progress_reset()
    set_progress("starting", 2, "شروع پردازش")
    ensure_backdrop()

    # --- Quicklook-based methods ---
    if method in ('kmeans_rgb', 'otsu_gray'):
        set_progress("load_quicklook", 5, "لود تصویر بک‌دراپ")
        img = Image.open(settings.BACKDROP_IMAGE).convert('RGB')
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape[:2]

        if method == 'kmeans_rgb':
            set_progress("kmeans_fit", 10, "آماده‌سازی KMeans (MiniBatch)")

            # اگر خواستید: ndvi_for_label = _compute_ndvi()[0]
            ndvi_for_label = None

            # ---- MiniBatchKMeans با گزارش پیشرفت هنگام predict روی چانک‌ها
            mask = _kmeans_mask_from_rgb(
                arr_rgb=arr,
                n_clusters=2,
                fit_max_side=2048,
                batch_size=16384,
                max_iter=100,
                random_state=0,
                predict_chunk_px=10_000_000,
                ndvi_for_label=ndvi_for_label,
                progress_cb=lambda phase, p: set_progress(phase, p)  # ⬅️ کال‌بک
            )

        else:  # otsu_gray
            set_progress("otsu_hist", 20, "محاسبه‌ی هیستوگرام سطح خاکستری")
            gray = (0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]).astype(np.uint8)

            hist, _ = np.histogram(gray, bins=256, range=(0, 255))
            total = int(hist.sum()); sumB = 0.0; wB = 0.0
            maximum = 0.0; sum1 = float(np.dot(np.arange(256), hist))
            threshold = 0
            for t in range(256):
                wB += hist[t]
                if wB == 0: 
                    continue
                wF = total - wB
                if wF == 0:
                    break
                sumB += t * hist[t]
                mB = sumB / wB
                mF = (sum1 - sumB) / wF
                between = wB * wF * (mB - mF) ** 2
                if between >= maximum:
                    threshold = t; maximum = between
                # آپدیت تدریجی (۲۰→۴۰٪)
                if t % 16 == 0:
                    frac = t / 255.0
                    set_progress("otsu_threshold", 20 + frac*20)

            mask = (gray <= threshold).astype(np.uint8)

        set_progress("save_mask", 85, "ذخیره ماسک و اوورلی")
        Image.fromarray(mask, mode='L').save(settings.MASK_PNG, optimize=False)
        from services.masks import write_mask_overlay
        write_mask_overlay(mask)

        set_progress("done", 100, "پایان")
        return True, 'ok'

    # --- NDVI-based methods ---
    elif method in ('ndvi_otsu', 'ndvi_thresh'):
        try:
            set_progress("read_bands", 8, "خواندن باندها (L2A)")
            ndvi, w_ndvi, h_ndvi = _compute_ndvi()
            set_progress("ndvi_computed", 35, "محاسبه NDVI")
        except Exception as e:
            set_progress("error", 100, f"NDVI failed: {e}")
            return False, f'NDVI failed: {e}'

        ndvi_vis = np.nan_to_num(ndvi, nan=-1.0, posinf=1.0, neginf=-1.0)
        ndvi_u8 = ((ndvi_vis + 1.0) * 127.5).astype(np.uint8)
        Image.fromarray(ndvi_u8, mode='L').save(settings.OUTPUT_DIR / "ndvi_preview.png")

        if method == 'ndvi_otsu':
            set_progress("otsu_on_ndvi", 45, "آستانه‌گذاری اوتسو روی NDVI")
            hist, _ = np.histogram(ndvi_u8, bins=256, range=(0, 255))
            total = int(hist.sum()); sumB = 0.0; wB = 0.0
            maximum = 0.0; sum1 = float(np.dot(np.arange(256), hist))
            threshold_idx = 0
            for t in range(256):
                wB += hist[t]
                if wB == 0: 
                    continue
                wF = total - wB
                if wF == 0:
                    break
                sumB += t * hist[t]
                mB = sumB / wB
                mF = (sum1 - sumB) / wF
                between = wB * wF * (mB - mF) ** 2
                if between >= maximum:
                    threshold_idx = t; maximum = between
                if t % 16 == 0:
                    set_progress("otsu_on_ndvi", 45 + (t/255.0)*10)
            thr = (threshold_idx / 255.0) * 2.0 - 1.0
        else:
            thr = float(kwargs.get('ndvi_threshold', getattr(settings, 'NDVI_DEFAULT_THRESHOLD', 0.2)))
            set_progress("thresholding", 55, f"آستانه ثابت NDVI={thr:.2f}")

        mask = (ndvi >= thr).astype(np.uint8)
        set_progress("mask_ready", 65, "ماسک ساخته شد")

        Image.fromarray((mask*255).astype(np.uint8), mode='L').save(settings.OUTPUT_DIR / "mask_vis_debug.png")

        if settings.BACKDROP_IMAGE.exists():
            W, H = Image.open(settings.BACKDROP_IMAGE).size
            if (w_ndvi, h_ndvi) != (W, H):
                set_progress("resize", 75, "هم‌اندازه‌سازی با بک‌دراپ")
                mask = np.array(Image.fromarray(mask, mode='L').resize((W, H), Image.NEAREST))

        set_progress("save", 85, "ذخیره mask.png و overlay")
        Image.fromarray(mask, mode='L').save(settings.MASK_PNG, optimize=False)
        from services.masks import write_mask_overlay
        write_mask_overlay(mask)

        set_progress("done", 100, "پایان")
        return True, 'ok'

    else:
        set_progress("error", 100, "روش نامعتبر")
        return False, 'unknown method'