from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class Settings:
    BASE_DIR: Path = Path(__file__).resolve().parent
    OUTPUT_DIR: Path = BASE_DIR / 'output'

    # Backdrop PNG; if missing and JP2 dir is set, it will be generated
    BACKDROP_IMAGE: Path = OUTPUT_DIR / 'rgb_quicklook.png'
    S2_RGB_TIF: Path = OUTPUT_DIR / 's2_rgb.tif'

    # OPTIONAL: set to your local JP2 10m directory (or leave empty)
    S2_JP2_DIR: Path = Path(r"/Users/ali/Downloads/FilesManualGroundTruth (1)/S2C_MSIL2A_20250607T070641_N0511_R106_T39RXN_20250607T105318.SAFE/GRANULE/L2A_T39RXN_A003937_20250607T071857/IMG_DATA/R10m")
    S2_BANDS: dict = field(default_factory=lambda: {"R": "B04", "G": "B03", "B": "B02", "NIR": "B08"})

    # Files
    POLYGONS_GEOJSON: Path = OUTPUT_DIR / 'polygons.geojson'
    POLYGONS_SHP: Path = OUTPUT_DIR / 'polygons.shp'
    MASK_PNG: Path = OUTPUT_DIR / 'mask.png'  # uint8, single channel, lossless

    # Classes shown in UI (mask stores numeric ids only)
    CLASS_LIST: list = field(default_factory=lambda: [
    {"name": "Background", "id": 0, "color": "#000000"},
    {"name": "Vegetation", "id": 1, "color": "#00ff00"},
    {"name": "Other", "id": 2, "color": "#8b4513"},
    ])

    DEFAULT_BRUSH_SIZE: int = 16
    DEFAULT_BRUSH_SHAPE: str = 'circle'  # or 'square'

    NDVI_DEFAULT_THRESHOLD: float = 0.2

    BOA_ADD_OFFSET = -1000.0     # از MTD_MSIL2A
    BOA_QUANT = 10000.0          # از MTD_MSIL2A
    BAD_DN_VALUES = (0, 65535)   # 0=NoData, 65535=Saturated
    NDVI_DEFAULT_THRESHOLD = 0.2
    
    MODELS_DIR         = OUTPUT_DIR / "models"
    ACTIVE_MODEL_PATH  = MODELS_DIR / "active.onnx"   # یا None
    
    MODEL_TYPE         = "onnx"       # or "torch"
    MODEL_BANDS        = ["B02","B03","B04","B08"]    # ترتیب ورودی مدل
    MODEL_INPUT_SIZE   = 256
    MODEL_NUM_CLASSES  = 3            # مثلا: 0=bg,1=veg,2=others
    MODEL_MEAN         = [0.3, 0.3, 0.3, 0.3]   # مثال؛ خودت طبق آموزش تنظیم کن
    MODEL_STD          = [0.2, 0.2, 0.2, 0.2]   # مثال
    MODEL_OVERLAP      = 32
    MODEL_BATCH_TILES  = 8            # اگر RAM اجازه می‌دهد
    
    USE_SCL_MASK      = False
    SCL_BAD_CLASSES   = [8, 9, 10]  # 8=Cloud, 9=Shadow, 10=Snow/Ice

settings = Settings()