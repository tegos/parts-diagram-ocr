"""Project paths and detection parameters (Python 3, pathlib-based)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
OUT_DIR = DATA_DIR / "out"

# --- glyph candidate filter (connected components, absolute px) ---
GLYPH_MIN_H, GLYPH_MAX_H = 16, 70     # single-digit glyph height
GLYPH_MIN_W, GLYPH_MAX_W = 3, 60      # was 6; 3 admits the narrow "1" digit (w4-5
                                      # with its serif/flag), which reads as a clean
                                      # callout. Solid 1px leader bars are still cut
                                      # by GLYPH_MAX_FILL; tall-thin line fragments by
                                      # GLYPH_MAX_ASPECT. See docs/adr/0002.
GLYPH_MIN_FILL, GLYPH_MAX_FILL = 0.18, 0.85  # area/bbox: rejects lines & solid blobs
GLYPH_MIN_AREA = 30          # was 50; a small "1" is only ~38px, so 50 dropped it
GLYPH_MAX_ASPECT = 7         # h/w cap. A printed "1" sits at ~3-4 at every scale
                             # (serifs scale with height); leader-line fragments that
                             # read as "1" are bare bars at aspect 10+ -> rejected.

# --- number grouping ---
GROUP_GAP = 0.8        # max horizontal gap between glyphs, in units of glyph height
GROUP_ROW_TOL = 0.6    # vertical centre tolerance, in units of glyph height

# --- whitespace-isolation gate ---
# Real callouts sit in white margin; drawing-internal blobs are surrounded by ink.
ISO_RING = 0.6         # ring thickness around the number box, in units of its height
ISO_MAX_INK = 0.10     # max black-pixel ratio in the ring. Real digits measured up
                       # to ~0.09; part features like a hex nut read as a digit sit
                       # higher (~0.10+), so this trims them. Tight margin — tuned
                       # against the eval set, was 0.12.

# --- recognition ---
RECOG_PAD = 6          # px padding around crop before recognition
RECOG_UPSCALE = 4.0    # upscale crop for the recognizer
MIN_CONFIDENCE = 0.55  # drop low-confidence reads
DIGIT_RANGE = (1, 199)  # numeric part of a callout; some catalogs exceed 100
ALLOWLIST = "0123456789AB"  # callouts can carry an A/B suffix (1A, 4A, 16A, 1B)
SUFFIXES = "AB"        # accepted trailing letters
NMS_IOU = 0.5          # dedup overlapping detections

DEBUG = True
