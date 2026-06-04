"""Per-image callout detection pipeline: CC detect -> gate -> group -> recognize."""
import cv2

from src import config as C
from src import glyphs as G
from src.ocr import recognize


def valid_number(text):
    if not text or not text.isdigit():
        return False
    if len(text) > 1 and text[0] == "0":   # no leading-zero callouts
        return False
    return C.DIGIT_RANGE[0] <= int(text) <= C.DIGIT_RANGE[1]


def detect(gray):
    """Return a list of detections: {text, conf, bbox:[x0,y0,x1,y1]}."""
    binv = G.binarize_inv(gray)
    cand = G.glyph_candidates(binv)
    groups = G.group_numbers(cand)

    dets = []
    for box in groups:
        if G.ink_ratio_ring(binv, box) > C.ISO_MAX_INK:   # whitespace gate
            continue
        text, conf = recognize(gray, box)
        if conf < C.MIN_CONFIDENCE or not valid_number(text):
            continue
        x, y, w, h = box
        dets.append({"text": text, "conf": round(conf, 3),
                     "bbox": [x, y, x + w, y + h]})

    return G.nms(dets, C.NMS_IOU)


def draw_overlay(bgr, dets):
    img = bgr.copy()
    for d in dets:
        x0, y0, x1, y1 = d["bbox"]
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(img, d["text"], (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
    return img
