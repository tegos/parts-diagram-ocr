"""EasyOCR wrapper — recognition only (detection is done by connected components)."""
import cv2

from src.config import RECOG_PAD, RECOG_UPSCALE

_reader = None


def reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def recognize(gray, box):
    """Read a digit string from a single number bbox (x, y, w, h).

    Returns (text, confidence); text is None if nothing digit-like is read.
    """
    x, y, w, h = box
    p = RECOG_PAD
    crop = gray[max(0, y - p):y + h + p, max(0, x - p):x + w + p]
    if crop.size == 0:
        return None, 0.0
    crop = cv2.resize(crop, None, fx=RECOG_UPSCALE, fy=RECOG_UPSCALE,
                      interpolation=cv2.INTER_CUBIC)
    res = reader().readtext(crop, allowlist="0123456789", detail=1,
                            text_threshold=0.4, low_text=0.3, mag_ratio=1.5)
    if not res:
        return None, 0.0
    _, text, conf = max(res, key=lambda r: r[2])
    return (text, float(conf)) if text.isdigit() else (None, 0.0)
