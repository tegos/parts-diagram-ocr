"""EasyOCR wrapper — recognition only (detection is done by connected components)."""
from src.config import ALLOWLIST, RECOG_PAD

_reader = None


def reader():
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def _clean(text):
    return text.upper() if text else None


def recognize_batch(gray, boxes):
    """Read digit strings for many number boxes in one batched call.

    boxes: list of (x, y, w, h). Returns a list of (text, conf) aligned with
    boxes; text is None where nothing digit-like is read. Batching collapses
    EasyOCR's huge per-call overhead (~450ms -> a few ms per box).
    """
    if not boxes:
        return []
    p = RECOG_PAD
    H, W = gray.shape[:2]
    h_list = [[max(0, x - p), min(W, x + w + p), max(0, y - p), min(H, y + h + p)]
              for (x, y, w, h) in boxes]
    res = reader().recognize(gray, h_list, [], allowlist=ALLOWLIST,
                             batch_size=64, detail=1)
    if len(res) == len(boxes):
        return [(_clean(t), float(c)) for (_b, t, c) in res]
    # rare misalignment: fall back to per-box reads
    return [recognize(gray, b) for b in boxes]


def recognize(gray, box):
    """Single-box read (fallback / spikes). See recognize_batch for the fast path."""
    import cv2
    from src.config import RECOG_UPSCALE
    x, y, w, h = box
    p = RECOG_PAD
    crop = gray[max(0, y - p):y + h + p, max(0, x - p):x + w + p]
    if crop.size == 0:
        return None, 0.0
    crop = cv2.resize(crop, None, fx=RECOG_UPSCALE, fy=RECOG_UPSCALE,
                      interpolation=cv2.INTER_CUBIC)
    out = reader().readtext(crop, allowlist=ALLOWLIST, detail=1,
                            text_threshold=0.4, low_text=0.3, mag_ratio=1.5)
    if not out:
        return None, 0.0
    _, text, conf = max(out, key=lambda r: r[2])
    return (_clean(text), float(conf))
