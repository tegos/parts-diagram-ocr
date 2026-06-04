"""Project paths and detection parameters (Python 3, pathlib-based)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_DIR = DATA_DIR / "samples"
OUT_DIR = DATA_DIR / "out"

# Detection / OCR params — tuned during Phase 1 spike.
UPSCALE = 2.0            # callout digits are small; upscale before OCR
MIN_CONFIDENCE = 0.30    # drop low-confidence OCR boxes
DIGIT_RANGE = (1, 99)    # callouts are part-index numbers, not part codes

DEBUG = True
