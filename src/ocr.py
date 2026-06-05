"""EasyOCR wrapper — recognition only (detection is done by connected components)."""
from src.config import ALLOWLIST, GLYPH_MIN_AREA, RECOG_PAD

_reader = None


def reader():
    global _reader
    if _reader is None:
        import os

        import easyocr
        import torch

        # torch defaults to 2 intraop threads here; use every core. CPU
        # inference is the whole pipeline cost (97% of batch wall time).
        torch.set_num_threads(os.cpu_count() or 2)
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


def recognize_split(gray, binv, box, max_glyphs=4):
    """Read a number box one glyph at a time and concatenate, as a fallback.

    EasyOCR can misread a digit when an adjacent suffix biases it: an open-top
    "4" next to an "A" reads as "AA", while each glyph in isolation reads
    cleanly. Splits the box into its connected components (left-to-right) and
    recognizes each alone. Returns (None, 0.0) when splitting can't help — a
    single component (whole == part) or more glyphs than a callout can hold
    (3 digits + 1 suffix). Group confidence is the weakest glyph's.
    """
    import cv2
    x, y, w, h = box
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv[y:y + h, x:x + w],
                                                      connectivity=8)
    ccs = []
    for i in range(1, n):
        cx, cy, cw, ch, area = (int(v) for v in stats[i])
        if area < GLYPH_MIN_AREA:        # drop speckle, keep glyph blobs
            continue
        ccs.append((x + cx, y + cy, cw, ch))
    if not (2 <= len(ccs) <= max_glyphs):
        return None, 0.0
    ccs.sort(key=lambda b: b[0])
    parts = [recognize(gray, b) for b in ccs]
    # each component must read as exactly one character; a multi-char read means
    # the blob isn't a clean single glyph (a narrow "1" can read as "11"), so the
    # split is unreliable -- bail rather than fabricate a longer number.
    if any(t is None or len(t) != 1 for t, _ in parts):
        return None, 0.0
    return _clean("".join(t for t, _ in parts)), min(c for _, c in parts)


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
    ch, cw = crop.shape[:2]
    out = reader().recognize(crop, [[0, cw, 0, ch]], [],
                             allowlist=ALLOWLIST, detail=1)
    if not out:
        return None, 0.0
    _, text, conf = max(out, key=lambda r: r[2])
    return (_clean(text), float(conf))
