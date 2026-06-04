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


def draw_overlay(bgr, dets, groups=None, open_braces=None):
    img = bgr.copy()
    # diagonal open-line braces: drawn (cyan) but not numbered -- see ADR 0002
    for (bx, by, bw, bh) in open_braces or []:
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)
        cv2.putText(img, "brace", (bx, by - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 200, 0), 2)
    for d in dets:
        x0, y0, x1, y1 = d["bbox"]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, d["text"], (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
    for g in groups or []:
        bx0, by0, bx1, by1 = g["brace_bbox"]
        cv2.rectangle(img, (bx0, by0), (bx1, by1), (255, 0, 255), 2)
        label = f"grp {g['group']}" if g["group"] else "grp ?"
        cv2.putText(img, label, (bx0, by0 - 6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 255), 2)
    return img
