"""Grouping-brace detection and callout association (Phase 3).

ETKA diagrams group a range of part numbers with a brace, in EITHER orientation:
- vertical '{'  : group-id beside it, members in the adjacent column
- horizontal '⏝': group-id above/below, members in the adjacent row

Detection finds elongated, thin, whitespace-isolated blobs. Association is
axis-agnostic: the side of the brace holding the most aligned callouts is the
member side; the lone callout near the brace tip on the opposite side is the id.
"""
import cv2

from src.glyphs import binarize_inv, ink_ratio_ring

# brace shape filter (elongated thin curve)
BRACE_MIN_LONG = 60        # length along the spine
BRACE_MIN_ASPECT = 3.0     # long / short
BRACE_MAX_SHORT = 80
BRACE_MAX_FILL = 0.35
BRACE_ISO_MAX_INK = 0.18   # whitespace gate; rejects part-contour lines

# association tolerances
GROUP_MIDBAND = 0.30       # group-id near brace tip (fraction of span length)
MEMBER_REACH = 0.13        # member row/column distance (fraction of perp image dim)
GROUP_REACH = 0.10         # group-id distance from spine (fraction of perp image dim)


def detect_braces(gray, binv=None):
    """Elongated, thin, whitespace-isolated blobs -> brace boxes (x, y, w, h)."""
    if binv is None:
        binv = binarize_inv(gray)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        long, short = max(w, h), min(w, h)
        if long < BRACE_MIN_LONG or short > BRACE_MAX_SHORT:
            continue
        if long / float(short) < BRACE_MIN_ASPECT:
            continue
        if area / float(w * h) > BRACE_MAX_FILL:
            continue
        if ink_ratio_ring(binv, (x, y, w, h)) > BRACE_ISO_MAX_INK:
            continue
        out.append((x, y, w, h))
    return out


def _cx(b):
    return (b["bbox"][0] + b["bbox"][2]) / 2.0


def _cy(b):
    return (b["bbox"][1] + b["bbox"][3]) / 2.0


def associate(braces, dets, image_width, image_height):
    """Return groups: {group, group_bbox, brace_bbox, members:[...]}."""
    groups = []
    for (bx, by, bw, bh) in braces:
        horizontal = bw >= bh
        if horizontal:
            span0, span1, span_len = bx, bx + bw, bw
            along, perp, perp_c, tip = _cx, _cy, by + bh / 2.0, bx + bw / 2.0
            member_reach = MEMBER_REACH * image_height
            group_reach = GROUP_REACH * image_height
        else:
            span0, span1, span_len = by, by + bh, bh
            along, perp, perp_c, tip = _cy, _cx, bx + bw / 2.0, by + bh / 2.0
            member_reach = MEMBER_REACH * image_width
            group_reach = GROUP_REACH * image_width

        pos, neg = [], []
        for d in dets:
            if not (span0 <= along(d) <= span1):
                continue
            off = perp(d) - perp_c
            (pos if off > 0 else neg).append((abs(off), d))

        pos_m = [d for o, d in pos if o <= member_reach]
        neg_m = [d for o, d in neg if o <= member_reach]
        if len(pos_m) >= len(neg_m):
            members, group_pool = pos_m, neg
        else:
            members, group_pool = neg_m, pos

        band = GROUP_MIDBAND * span_len
        cand = [d for o, d in group_pool
                if o <= group_reach and abs(along(d) - tip) <= band]
        group = min(cand, key=lambda d: abs(along(d) - tip)) if cand else None

        if not members or group is None:   # a group needs both an id and members
            continue
        groups.append({
            "group": group["text"],
            "group_bbox": group["bbox"],
            "brace_bbox": [bx, by, bx + bw, by + bh],
            "members": [{"text": d["text"], "bbox": d["bbox"]} for d in members],
        })
    return groups
