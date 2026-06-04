"""Grouping-brace detection and callout association (Phase 3).

ETKA diagrams use a tall '{' brace to group a range of part numbers: a group-id
callout sits on the brace's outer (spine) side, the member callouts sit within
its vertical span on the inner (drawing) side.
"""
import cv2

from src import config as C
from src.glyphs import binarize_inv, ink_ratio_ring

# brace shape filter
BRACE_MIN_H = 60
BRACE_MIN_ASPECT = 3.0
BRACE_MAX_W = 80
BRACE_MAX_FILL = 0.35
BRACE_ISO_MAX_INK = 0.18   # whitespace gate (looser than digits: brace curve adds ink)

# association tolerances
GROUP_MIDBAND = 0.30       # group-id band near brace mid-height (fraction of brace h)
MEMBER_MAX_DX = 0.13       # members sit in the column next to the brace (fraction of W)
GROUP_MAX_DX = 0.10        # group-id sits just outside the brace (fraction of W)


def detect_braces(gray, binv=None):
    """Tall, thin, whitespace-isolated blobs -> brace boxes (x, y, w, h)."""
    if binv is None:
        binv = binarize_inv(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if h < BRACE_MIN_H or w > BRACE_MAX_W:
            continue
        if h / float(w) < BRACE_MIN_ASPECT:
            continue
        if area / float(w * h) > BRACE_MAX_FILL:
            continue
        if ink_ratio_ring(binv, (x, y, w, h)) > BRACE_ISO_MAX_INK:
            continue  # reject part-contour verticals inside the drawing
        out.append((x, y, w, h))
    return out


def _cx(b):  # detection bbox centre x
    return (b["bbox"][0] + b["bbox"][2]) / 2.0


def _cy(b):
    return (b["bbox"][1] + b["bbox"][3]) / 2.0


def associate(braces, dets, image_width):
    """Return groups: {group, group_bbox, brace_bbox, members:[...]}.

    Each brace: opens toward image centre. The group-id callout is on the outer
    side near mid-height; members are inner-side callouts within the vertical span.
    """
    member_dx = MEMBER_MAX_DX * image_width
    group_dx = GROUP_MAX_DX * image_width
    groups = []
    for (bx, by, bw, bh) in braces:
        mid_y = by + bh / 2.0
        bx_c = bx + bw / 2.0                       # split on the brace centre:
        opens_right = bx_c < image_width / 2.0     # the number column straddles the edge

        inner, outer = [], []
        for d in dets:
            if not (by <= _cy(d) <= by + bh):
                continue
            off = _cx(d) - bx_c                     # +ve = right of brace
            dx = off if opens_right else -off       # signed toward the inner side
            if dx > 0:
                if dx <= member_dx:                 # the adjacent column only
                    inner.append(d)
            else:
                outer.append(d)

        # group-id: outer-side detection near mid-height, just outside the spine
        band = GROUP_MIDBAND * bh
        cand = [d for d in outer
                if abs(_cy(d) - mid_y) <= band and abs(_cx(d) - bx_c) <= group_dx]
        group = min(cand, key=lambda d: abs(_cx(d) - bx_c)) if cand else None

        if not inner:
            continue
        groups.append({
            "group": group["text"] if group else None,
            "group_bbox": group["bbox"] if group else None,
            "brace_bbox": [bx, by, bx + bw, by + bh],
            "members": [{"text": d["text"], "bbox": d["bbox"]} for d in inner],
        })
    return groups
