"""Connected-components glyph detection, grouping, and whitespace gating.

Pure-ish CV helpers; the recognition step lives in src.ocr.
"""
import cv2
import numpy as np

from src import config as C


def binarize_inv(gray):
    """Black ink -> white foreground (255), background 0."""
    _, binv = cv2.threshold(gray, 0, 255,
                            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return binv


def _linearity(labels, idx, box):
    """Max perpendicular deviation of component pixels from their PCA line,
    normalized by the box long side. A bare bar -> ~stroke_width/length; a
    digit (even a narrow serifed "1") deviates an order of magnitude more."""
    x, y, w, h = box
    ys, xs = np.nonzero(labels[y:y + h, x:x + w] == idx)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    q = pts - pts.mean(0)
    _, _, vt = np.linalg.svd(q, full_matrices=False)
    return float(np.abs(q @ vt[1]).max()) / max(w, h)


def glyph_candidates(binv):
    """Single-digit-shaped blobs as (x, y, w, h)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if not (C.GLYPH_MIN_H <= h <= C.GLYPH_MAX_H):
            continue
        if not (C.GLYPH_MIN_W <= w <= C.GLYPH_MAX_W):
            continue
        if area < C.GLYPH_MIN_AREA:
            continue
        if h / float(w) > C.GLYPH_MAX_ASPECT:   # reject tall-thin line fragments
            continue
        fill = area / float(w * h)
        if not (C.GLYPH_MIN_FILL <= fill <= C.GLYPH_MAX_FILL):
            continue
        # tilted bare bars evade the aspect cap (tilt inflates w); a straight
        # line is not a digit, however it leans
        if h / float(w) >= 3 and _linearity(labels, i, (x, y, w, h)) < C.GLYPH_MAX_LINEARITY:
            continue
        out.append((x, y, w, h))
    return out


def group_numbers(boxes):
    """Merge horizontally-adjacent glyphs of similar height into number boxes."""
    # sort by x only: merging proceeds rightward, so the leftmost glyph of a
    # number must be visited first. (A y-bin presort split same-row digits
    # whose y values straddled a bin boundary -- "14" became "1" + "4".)
    boxes = sorted(boxes, key=lambda b: b[0])
    used = [False] * len(boxes)
    groups = []
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        x, y, w, h = b
        used[i] = True
        right, cy = x + w, y + h // 2
        merged = True
        while merged:
            merged = False
            for j, c in enumerate(boxes):
                if used[j]:
                    continue
                jx, jy, jw, jh = c
                same_row = abs((jy + jh // 2) - cy) < C.GROUP_ROW_TOL * h
                close = 0 <= (jx - right) < C.GROUP_GAP * h
                similar = 0.6 < jh / h < 1.7
                if same_row and close and similar:
                    used[j] = True
                    right = max(right, jx + jw)
                    ny = min(y, jy)
                    h = max(y + h, jy + jh) - ny
                    y = ny
                    merged = True
        groups.append((x, y, right - x, h))
    return groups


def ink_ratio_ring(binv, box):
    """Fraction of ink in a ring around the box, excluding the box itself.

    Low value => isolated in whitespace (a real callout); high => inside drawing.
    """
    x, y, w, h = box
    r = int(C.ISO_RING * h)
    H, W = binv.shape
    ox0, oy0 = max(0, x - r), max(0, y - r)
    ox1, oy1 = min(W, x + w + r), min(H, y + h + r)
    outer = binv[oy0:oy1, ox0:ox1]
    if outer.size == 0:
        return 1.0
    ink_outer = int((outer > 0).sum())
    ink_inner = int((binv[y:y + h, x:x + w] > 0).sum())
    ring_pixels = outer.size - (w * h)
    if ring_pixels <= 0:
        return 1.0
    return (ink_outer - ink_inner) / float(ring_pixels)


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / float(area_a + area_b - inter)


def nms(dets, iou_thr):
    """Keep highest-confidence detection among overlapping ones."""
    kept = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(iou(d["bbox"], k["bbox"]) < iou_thr for k in kept):
            kept.append(d)
    return kept
