from __future__ import annotations
import json
import geopandas as gpd
from config import settings


def save_polygons_fc(fc: dict):
    if not fc or fc.get('type') != 'FeatureCollection':
        return False, 'invalid geojson'

    feats = []
    for i, f in enumerate(fc.get('features', [])):
        if 'properties' not in f:
            f['properties'] = {}
        if 'uid' not in f['properties']:
            f['properties']['uid'] = f"poly_{i+1:06d}"
        if 'class_id' not in f['properties']:
            col = (f['properties'].get('color') or '#00ff00').lower()
            class_id = 1 if col == '#00ff00' else 2 if col == '#8b4513' else 0
            f['properties']['class_id'] = class_id
        feats.append(f)

    out_fc = {"type": "FeatureCollection", "features": feats}
    settings.POLYGONS_GEOJSON.write_text(json.dumps(out_fc, ensure_ascii=False))

    try:
        gdf = gpd.GeoDataFrame.from_features(out_fc)
        if not gdf.empty:
            gdf.to_file(settings.POLYGONS_SHP, driver='ESRI Shapefile')
    except Exception as e:
        print('[WARN] shapefile save failed:', e)

    return True, 'ok'