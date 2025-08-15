from __future__ import annotations
from pathlib import Path  # ✅ برای Path(...)
import numpy as np        # ✅ برای np.unique
from flask import Blueprint, current_app, jsonify, request, send_from_directory, make_response

from config import settings
from services.s2 import ensure_backdrop, backdrop_meta, s2_bounds_wgs84, prelabel
from services.masks import mask_bytes, save_mask_bytes, load_mask
from services.polygons import save_polygons_fc
from services.progress import get_progress, set_progress, reset
from services import model as model_srv

api_bp = Blueprint('api', __name__)

# ---------- static-like route for output files ----------
@api_bp.route('/output/<path:filename>')
def output_files(filename):
    return send_from_directory(current_app.config['OUTPUT_DIR'], filename)

# ---------- backdrop / bounds ----------
@api_bp.route('/backdrop_meta')
def api_backdrop_meta():
    w, h = backdrop_meta()
    return jsonify({"width": w, "height": h})

@api_bp.route('/s2_bounds_wgs84')
def api_s2_bounds_wgs84():
    b = s2_bounds_wgs84()
    if b is None:
        return jsonify({"error": "s2_rgb.tif not found"}), 404
    return jsonify(b)

# ---------- polygons ----------
@api_bp.route('/save_polygons', methods=['POST'])
def api_save_polygons():
    fc = request.get_json(force=True, silent=True)
    ok, msg = save_polygons_fc(fc)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True})

# ---------- mask raw / save ----------
@api_bp.route('/mask_raw')
def api_mask_raw():
    ensure_backdrop()
    w, h = backdrop_meta()
    b = mask_bytes(w, h)
    resp = make_response(b)
    resp.headers['Content-Type'] = 'application/octet-stream'
    resp.headers['Cache-Control'] = 'no-store'
    return resp

@api_bp.route('/save_mask', methods=['POST'])
def api_save_mask():
    raw = request.get_data()
    w, h = backdrop_meta()
    ok, msg = save_mask_bytes(raw, w, h)
    if not ok:
        return jsonify({"error": msg}), 400
    return jsonify({"ok": True})

# ---------- prelabel ----------
@api_bp.route('/prelabel', methods=['POST'])
def api_prelabel():
    # اختیاری: هر بار که کار جدید شروع می‌شود، progress ریست شود
    reset()
    set_progress("starting", 2, "شروع پیش‌برچسب‌گذاری")

    body = request.get_json(force=True, silent=True) or {}
    method = (body.get('method') or 'kmeans_rgb').strip()

    kwargs = {}
    if method == 'ndvi_thresh':
        try:
            kwargs['ndvi_threshold'] = float(body.get('ndvi_threshold', settings.NDVI_DEFAULT_THRESHOLD))
        except Exception:
            kwargs['ndvi_threshold'] = settings.NDVI_DEFAULT_THRESHOLD

    ok, msg = prelabel(method, **kwargs)
    if not ok:
        set_progress("error", 100, str(msg))
        return jsonify({"error": msg}), 400

    set_progress("done", 100, "اتمام پیش‌برچسب‌گذاری")
    return jsonify({"ok": True})

# ---------- mask stats ----------
@api_bp.route('/mask_stats')
def mask_stats():
    w, h = backdrop_meta()
    m = load_mask(w, h)
    vals, cnts = np.unique(m, return_counts=True)
    return jsonify({
        'width': w, 'height': h,
        'counts': {int(v): int(c) for v, c in zip(vals, cnts)}
    })

# ---------- progress ----------
@api_bp.route('/progress')
def api_progress():
    return jsonify(get_progress())

# ---------- model endpoints ----------
@api_bp.route('/model_info')
def api_model_info():
    try:
        info = model_srv.model_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({"loaded": False, "error": str(e)}), 500

@api_bp.route('/model_upload', methods=['POST'])
def api_model_upload():
    # حالت 1: آپلود فایل (multipart/form-data)
    f = request.files.get('file')
    if f:
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        save_path = settings.MODELS_DIR / f.filename
        f.save(save_path)
        settings.ACTIVE_MODEL_PATH = save_path
    else:
        # حالت 2: JSON با مسیر فایل
        data = request.get_json(silent=True) or {}
        p = data.get('path')
        if not p:
            return jsonify({"error": "no model file or path supplied"}), 400
        settings.ACTIVE_MODEL_PATH = Path(p)

    info = model_srv.load_model(settings.ACTIVE_MODEL_PATH)
    return jsonify({"ok": True, "info": info})

@api_bp.route('/run_model', methods=['POST'])
def api_run_model():
    try:
        reset()
        set_progress("starting", 1, "شروع استنتاج مدل")
        ok, msg = model_srv.run_model_inference()
        if not ok:
            set_progress("error", 100, str(msg))
            return jsonify({"error": msg}), 400
        set_progress("done", 100, "اتمام استنتاج مدل")
        return jsonify({"ok": True})
    except Exception as e:
        set_progress("error", 100, str(e))
        return jsonify({"error": str(e)}), 500