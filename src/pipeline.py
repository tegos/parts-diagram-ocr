"""Per-image callout detection pipeline: CC detect -> gate -> group -> recognize."""
import cv2

from src import config as C
from src import glyphs as G
from src.ocr import recognize_batch


import re

_CALLOUT = re.compile(rf"^(\d{{1,3}})([{C.SUFFIXES}]?)$")


def valid_callout(text):
    """Accept N or N+suffix (e.g. 7, 29, 1A, 16B); numeric part within range."""
    if not text:
        return False
    m = _CALLOUT.match(text)
    if not m:
        return False
    num = m.group(1)
    if len(num) > 1 and num[0] == "0":   # no leading-zero callouts
        return False
    return C.DIGIT_RANGE[0] <= int(num) <= C.DIGIT_RANGE[1]


def detect(gray):
    """Return a list of detections: {text, conf, bbox:[x0,y0,x1,y1]}."""
    binv = G.binarize_inv(gray)
    cand = G.glyph_candidates(binv)
    groups = G.group_numbers(cand)
    gated = [b for b in groups if G.ink_ratio_ring(binv, b) <= C.ISO_MAX_INK]

    dets = []
    for box, (text, conf) in zip(gated, recognize_batch(gray, gated)):
        if conf < C.MIN_CONFIDENCE or not valid_callout(text):
            continue
        x, y, w, h = box
        dets.append({"text": text, "conf": round(conf, 3),
                     "bbox": [x, y, x + w, y + h]})

    return G.nms(dets, C.NMS_IOU)


_FONT = cv2.FONT_HERSHEY_SIMPLEX
OVERLAY_ALPHA = 0.7        # annotation opacity; untouched pixels stay fully opaque


def _callout(layer, anchor, text, color, scale=0.6, thick=2, dx=16, dy=-18):
    """Draw `text` as an offset pill linked to `anchor` by a thin leader line."""
    ax, ay = anchor
    H, W = layer.shape[:2]
    (tw, th), base = cv2.getTextSize(text, _FONT, scale, thick)
    lx = min(max(ax + dx, 2), W - tw - 6)        # keep the pill on-canvas
    ly = min(max(ay + dy, th + 6), H - base - 2)
    cv2.line(layer, (ax, ay), (lx, ly - th // 2), color, 1, cv2.LINE_AA)
    cv2.rectangle(layer, (lx - 4, ly - th - 4), (lx + tw + 4, ly + base), color, -1)
    cv2.putText(layer, text, (lx, ly), _FONT, scale, (255, 255, 255),
                thick, cv2.LINE_AA)


def draw_overlay(bgr, dets, groups=None, open_braces=None, alpha=OVERLAY_ALPHA):
    """Annotations are drawn on a layer and alpha-blended, so each mark is
    semi-transparent while untouched pixels keep the original drawing intact.
    Detected numbers are pulled out as small leader-linked callouts."""
    layer = bgr.copy()
    # diagonal open-line braces: drawn (cyan) but not numbered -- see ADR 0002
    for (bx, by, bw, bh) in open_braces or []:
        cv2.rectangle(layer, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
        _callout(layer, (bx, by), "brace", (255, 200, 0))
    for d in dets:
        x0, y0, x1, y1 = d["bbox"]
        cv2.rectangle(layer, (x0, y0), (x1, y1), (0, 0, 255), 1)
        _callout(layer, (x1, y0), d["text"], (0, 0, 255))
    for g in groups or []:
        bx0, by0, bx1, by1 = g["brace_bbox"]
        cv2.rectangle(layer, (bx0, by0), (bx1, by1), (255, 0, 255), 2)
        label = f"grp {g['group']}" if g["group"] else "grp ?"
        _callout(layer, (bx0, by0), label, (255, 0, 255), scale=0.7)
    return cv2.addWeighted(layer, alpha, bgr, 1 - alpha, 0)
