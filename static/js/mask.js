// ===============================
// mask.js  –  Canvas mask editor
// ===============================

// ----- Canvas refs -----
const backdrop = document.getElementById('backdrop');
const mask = document.getElementById('mask');
const ctxB = backdrop.getContext('2d');
const ctxM = mask.getContext('2d');

// ----- State -----
let tool = 'brush';
let zoom = 1;
let panX = 0, panY = 0;      // در واحد CSS px
let W = 0, H = 0;            // ابعاد تصویر پس‌زمینه (پیکسل تصویر)
let baseScale = 1;           // fit-to-view scale (CSS px per image px)

// نمایش ماسک روی offscreen
let maskImageData = null;    // RGBA نمایش اوورلی
let maskOffscreen = null, maskOffCtx = null;

// کلاس واقعی پیکسل‌ها (0=پس‌زمینه، 1=Vegetation، 2=Other)
let classMap = null;

// پالت رنگ برای نمایش (صرفاً ویژوال)
const PALETTE = {
  0: [0, 0, 0, 0],           // شفاف
  1: [0, 255, 0, 180],       // سبز نیمه‌شفاف
  2: [139, 69, 19, 180]      // قهوه‌ای نیمه‌شفاف
};

// ----- Backdrop (با cache-busting) -----
const imgB = new Image();
imgB.src = '/api/output/rgb_quicklook.png?t=' + Date.now();

// ----- Layout / Resize -----
function resizeCanvas(){
  const dpr = window.devicePixelRatio || 1;
  const rect = document.querySelector('.canvas-wrap').getBoundingClientRect();
  const cw = rect.width, ch = rect.height;

  // اندازه‌ی بافر کانواس‌ها به پیکسل دستگاه
  backdrop.width = cw * dpr; backdrop.height = ch * dpr;
  backdrop.style.width = cw + 'px'; backdrop.style.height = ch + 'px';

  mask.width = cw * dpr; mask.height = ch * dpr;
  mask.style.width = cw + 'px'; mask.style.height = ch + 'px';

  // محاسبه‌ی baseScale برای جای‌گذاری تصویر در قاب
  if (W > 0 && H > 0) {
    const sx = cw / W, sy = ch / H;
    baseScale = Math.min(sx, sy);
    const viewScale = baseScale * zoom;
    panX = (cw - W * viewScale) / 2;
    panY = (ch - H * viewScale) / 2;
  }

  draw();
}

window.addEventListener('resize', resizeCanvas);

// ----- Draw -----
function draw(){
  const rect = document.querySelector('.canvas-wrap').getBoundingClientRect();
  const cw = rect.width, ch = rect.height;
  const dpr = window.devicePixelRatio || 1;
  const viewScale = baseScale * zoom;

  // --- بک‌دراپ ---
  ctxB.setTransform(dpr, 0, 0, dpr, 0, 0); // کار در واحد CSS، نگاشت به پیکسل دستگاه
  ctxB.clearRect(0, 0, cw, ch);
  ctxB.save();
  ctxB.translate(panX, panY);
  ctxB.scale(viewScale, viewScale);
  ctxB.imageSmoothingEnabled = false;
  if (W && H) {
    ctxB.drawImage(imgB, 0, 0, W, H, 0, 0, W, H);
  }
  ctxB.restore();

  // --- ماسک ---
  ctxM.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctxM.clearRect(0, 0, cw, ch);
  ctxM.save();
  ctxM.translate(panX, panY);
  ctxM.scale(viewScale, viewScale);
  ctxM.imageSmoothingEnabled = false;
  ctxM.globalAlpha = 1.0; // آلفا داخل پالت
  if (maskOffscreen) {
    ctxM.drawImage(maskOffscreen, 0, 0, W, H, 0, 0, W, H);
  }
  ctxM.restore();
}

// ----- Build RGBA from classMap -----
function rebuildOverlayFromClassMap(){
  if (!classMap) return;
  const rgba = new Uint8ClampedArray(W * H * 4);
  for (let i = 0; i < W * H; i++) {
    const c = classMap[i] | 0;
    const p = PALETTE[c] || PALETTE[0];
    const o = i * 4;
    rgba[o+0] = p[0];
    rgba[o+1] = p[1];
    rgba[o+2] = p[2];
    rgba[o+3] = p[3];
  }
  maskImageData = new ImageData(rgba, W, H);
  maskOffCtx.putImageData(maskImageData, 0, 0);
}

// ----- Fetch mask (raw bytes) -----
async function fetchMask(){
  try{
    const r = await fetch('/api/mask_raw?t=' + Date.now(), { cache: 'no-store' });
    const buf = await r.arrayBuffer();
    const arr = new Uint8Array(buf);

    // اندازه را با W,H چک کن
    if (arr.length !== W*H) {
      console.warn('mask size mismatch', arr.length, 'vs', W*H, '→ using empty classMap sized W×H');
      classMap = new Uint8Array(W*H); // صفر
    } else {
      classMap = arr; // بدون کپی
    }

    // گزارش سریع
    let nz = 0; for (let i=0;i<classMap.length;i++) if (classMap[i]) nz++;
    console.log('mask non-zero:', nz, 'of', classMap.length);

    rebuildOverlayFromClassMap();
    draw();
  }catch(err){
    console.error('fetchMask failed', err);
  }
}

// ----- Image onload -----
imgB.onload = () => {
  W = imgB.naturalWidth;
  H = imgB.naturalHeight;

  // offscreen
  maskOffscreen = document.createElement('canvas');
  maskOffscreen.width = W;
  maskOffscreen.height = H;
  maskOffCtx = maskOffscreen.getContext('2d');

  // بعد از لود تصویر، یک بار resize برای محاسبه‌ی baseScale
  resizeCanvas();
  fetchMask();
};

// ----- Screen ↔ Image coords -----
function screenToImageFromEvent(e){
  // مختصات موس نسبت به خود canvas (CSS px)
  const rect = mask.getBoundingClientRect();
  const xCss = e.clientX - rect.left;
  const yCss = e.clientY - rect.top;

  const viewScale = baseScale * zoom;

  // تبدیل به مختصات تصویر (پیکسل تصویر)
  const ix = Math.floor((xCss - panX) / viewScale);
  const iy = Math.floor((yCss - panY) / viewScale);
  return [ix, iy];
}

// ----- Painting -----
function paintAt(ix, iy){
  if (!classMap || !maskImageData) return;

  const size = parseInt(document.getElementById('brushSize')?.value || 16);
  const shape = document.getElementById('brushShape')?.value || 'circle';
  const cid   = parseInt(document.getElementById('classId')?.value || 1); // پیش‌فرض 1

  const half = Math.floor(size/2);
  const data = maskImageData.data;
  const p = PALETTE[cid] || PALETTE[0];

  function setPx(x, y){
    if (x < 0 || y < 0 || x >= W || y >= H) return;
    const idx = y * W + x;
    classMap[idx] = cid; // داده‌ی واقعی
    const off = idx * 4; // داده‌ی نمایشی
    data[off+0] = p[0]; data[off+1] = p[1];
    data[off+2] = p[2]; data[off+3] = p[3];
  }

  for (let dy = -half; dy <= half; dy++) {
    for (let dx = -half; dx <= half; dx++) {
      const x = ix + dx, y = iy + dy;
      if (shape === 'circle' && (dx*dx + dy*dy) > half*half) continue;
      setPx(x, y);
    }
  }

  maskOffCtx.putImageData(maskImageData, 0, 0);
  draw();
}

// ----- Events -----
let isDown = false;

mask.addEventListener('mousedown', (e) => {
  isDown = true;
  mask.style.cursor = (tool === 'pan') ? 'grabbing' : 'crosshair';
  if (tool === 'brush') {
    const [ix, iy] = screenToImageFromEvent(e);
    paintAt(ix, iy);
  }
});

mask.addEventListener('mousemove', (e) => {
  if (!isDown) return;
  if (tool === 'brush') {
    const [ix, iy] = screenToImageFromEvent(e);
    paintAt(ix, iy);
  } else if (tool === 'pan') {
    // movementX/Y در CSS pixels
    panX += e.movementX;
    panY += e.movementY;
    draw();
  }
});

window.addEventListener('mouseup', () => {
  isDown = false;
  mask.style.cursor = (tool === 'pan') ? 'grab' : 'crosshair';
});

// زوم حول محل موس
mask.addEventListener('wheel', (e) => {
  if (tool !== 'zoom') return;
  e.preventDefault();

  const rect = mask.getBoundingClientRect();
  const xCss = e.clientX - rect.left;
  const yCss = e.clientY - rect.top;

  const scaleBefore = baseScale * zoom;
  const factor = e.deltaY < 0 ? 1.1 : 0.9;
  const scaleAfter = scaleBefore * factor;

  // حفظ نقطه‌ی موس روی همان نقطه‌ی تصویر
  const ix = (xCss - panX) / scaleBefore;
  const iy = (yCss - panY) / scaleBefore;

  zoom *= factor;

  panX = xCss - ix * scaleAfter;
  panY = yCss - iy * scaleAfter;

  draw();
}, { passive: false });

// ----- Toolbar -----
document.getElementById('toolBrush')?.addEventListener('click', () => {
  tool = 'brush'; mask.style.cursor = 'crosshair';
});
document.getElementById('toolPan')?.addEventListener('click',   () => {
  tool = 'pan';   mask.style.cursor = 'grab';
});
document.getElementById('toolZoom')?.addEventListener('click',  () => {
  tool = 'zoom';  mask.style.cursor = 'zoom-in';
});

// ----- Save (lossless) -----
document.getElementById('saveMaskBtn')?.addEventListener('click', async () => {
  if (!classMap) return;
  const r = await fetch('/api/save_mask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/octet-stream', 'Cache-Control': 'no-store' },
    body: classMap      // 0/1/2/...
  });
  alert(r.ok ? 'ماسک ذخیره شد' : 'خطا در ذخیره ماسک');
});