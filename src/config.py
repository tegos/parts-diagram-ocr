"""Project paths and detection parameters (Python 3, pathlib-based)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_DIR = DATA_DIR / "samples"
OUT_DIR = DATA_DIR / "out"

# --- glyph candidate filter (connected components, absolute px) ---
GLYPH_MIN_H, GLYPH_MAX_H = 16, 70     # single-digit glyph height
GLYPH_MIN_W, GLYPH_MAX_W = 6, 60      # single-digit glyph width
GLYPH_MIN_FILL, GLYPH_MAX_FILL = 0.18, 0.85  # area/bbox: rejects lines & solid blobs
GLYPH_MIN_AREA = 60

# --- number grouping ---
GROUP_GAP = 0.8        # max horizontal gap between glyphs, in units of glyph height
GROUP_ROW_TOL = 0.6    # vertical centre tolerance, in units of glyph height

# --- whitespace-isolation gate ---
# Real callouts sit in white margin; drawing-internal blobs are surrounded by ink.
ISO_RING = 0.6         # ring thickness around the number box, in units of its height
ISO_MAX_INK = 0.12     # max black-pixel ratio allowed in the ring

# --- recognition ---
RECOG_PAD = 6          # px padding around crop before recognition
RECOG_UPSCALE = 4.0    # upscale crop for the recognizer
MIN_CONFIDENCE = 0.55  # drop low-confidence reads
DIGIT_RANGE = (1, 99)  # callouts are part-index numbers
NMS_IOU = 0.5          # dedup overlapping detections

DEBUG = True
