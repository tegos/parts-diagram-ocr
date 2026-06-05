"""Grouping-brace detection and callout association (Phase 3).

Parts-catalog diagrams group a range of part numbers with a brace, in EITHER orientation:
- vertical '{'  : group-id beside it, members in the adjacent column
- horizontal '⏝': group-id above/below, members in the adjacent row

Detection finds elongated, thin, whitespace-isolated blobs. Association is
axis-agnostic: the brace's notch (cusp of '{', apex of V) points at the group
id; with no measurable notch, the side holding the most aligned callouts is
the member side and the id is the lone callout near the tip opposite.
"""
import cv2
import numpy as np

from src.glyphs import binarize_inv, ink_ratio_ring, iou

# brace shape filter (elongated thin curve)
BRACE_MIN_LONG = 60        # length along the spine
BRACE_MIN_ASPECT = 3.0     # long / short
BRACE_MAX_SHORT = 80
BRACE_MAX_FILL = 0.35
BRACE_ISO_MAX_INK = 0.18   # whitespace gate; rejects part-contour lines

# open-line brace filter (Phase 3b): grouping braces drawn as DIAGONAL thin lines.
# A diagonal line has a near-square bbox, so the aspect test above misses it. These
# braces are instead identified as thin (low-fill), whitespace-isolated, OPEN
# polylines -- holes==0 separates them from the closed loops of part contours.
# See docs/adr/0002. Detected for the overlay only; not auto-numbered into groups
# (the prong/id geometry is too noisy on flat exploded views -- ADR 0002).
OPEN_MIN_AREA = 150        # line length in px (thin line: area ~= length)
OPEN_MAX_AREA = 4000       # above this it's a part body, not a brace line
OPEN_MAX_FILL = 0.06       # area/bbox: a thin line nearly vanishes in its bbox
OPEN_MIN_LONG = 80         # bbox long side: span at least a couple of rows
OPEN_ISO_MAX_INK = 0.06    # whitespace gate, tighter than solid braces
OPEN_MAX_HOLES = 0         # open polyline encloses nothing; part outlines do
OPEN_MIN_CORNERS = 5       # sharp direction changes (prongs). A brace is a
                           # bracket -- prongs are its grouping semantics; a
                           # plain straight stroke is a leader line. Measured:
                           # real braces 7-76 corners, leader lines 1-3.
OPEN_MUTUAL_IOU = 0.35     # candidates whose bboxes overlap this much are
                           # fragments of one drawing entity (e.g. concentric
                           # flywheel arcs broken by teeth) -- a real brace
                           # stands alone. Measured: arc pairs 0.37-0.7+,
                           # nested-but-distinct real braces 0.015.

# association tolerances
GROUP_MIDBAND = 0.30       # group-id near brace tip (fraction of span length)
MEMBER_REACH = 0.13        # member row/column distance (fraction of perp image dim)
GROUP_REACH = 0.10         # group-id distance from spine (fraction of perp image dim)
NOTCH_MIN_DEPTH = 0.5      # min notch depth / median digit height to trust
                           # the notch side over the count-vote. Measured on
                           # the 20-image GT set (ADR 0002 id-swap update):
                           # true cusps/apexes 0.57-0.97 dh (swapped-id braces
                           # 0.75-0.97); nothing measurable below 0.57. The
                           # high floor is the safe direction -- dropping a
                           # true notch only reverts to the count-vote.
# diagonal-brace association tolerances, in units of DIGIT HEIGHT -- the only
# scale that holds across catalogs (image-dim fractions break: a 0.03*2300px
# member band swallows the id on large diagrams). Measured on every true brace
# (5 images, 12 braces): id sits 0.7-1.0 digit-heights from the prong tip
# (false bindings: 1.5-4.8), members hug the spine at 0.6-1.6 on the opposite
# side from the id (unrelated callouts: 8+).
OPEN_ID_REACH = 1.3        # max dist(prong tip, id centre) / id height
OPEN_MEMBER_REACH = 4.5    # max |spine offset| / median digit height for members
OPEN_MEMBER_SIDE = 1.0     # members may poke at most this far onto the id side
OPEN_SPAN_PAD = 10         # px slack at the spine's ends for the span test
OPEN_BIND_MIN_SHARP = 5    # central sharp vertices required to BIND a brace
                           # to a group. A true grouping bracket is pronged
                           # along its spine: every true diagonal brace on
                           # the 20-image GT set shows 6-9 sharp vertices in
                           # the central 80% of the span (sample.png: 6/6/10);
                           # the two false part-contour bindings (645845000
                           # "6", 258133860 "2") show exactly 3. Gate at 5.


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


def _hole_count(label_crop, idx):
    """Holes enclosed by component `idx` -- 0 for an open line, >0 for a loop."""
    mask = (label_crop == idx).astype("uint8") * 255
    _, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return 0
    return sum(1 for c in hier[0] if c[3] != -1)   # contour with a parent = hole


def _sharp_corners(labels, idx, box):
    """Sharp direction changes along component `idx`'s outline.

    Prongs of a bracket produce corners well under 140 deg; a straight leader
    line has none and a smooth arc only shallow ones. Short polygon edges
    (under 8 px) are stroke-width noise and don't vote.
    """
    x, y, w, h = box
    mask = (labels[y:y + h, x:x + w] == idx).astype("uint8")
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 4, False)[:, 0, :]
    pts = pts.astype(np.float64)
    sharp = 0
    for i in range(1, len(pts) - 1):
        v1, v2 = pts[i - 1] - pts[i], pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 8 or n2 < 8:
            continue
        ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1)))
        if ang < 140:
            sharp += 1
    return sharp


def _xyxy(b):
    x, y, w, h = b
    return (x, y, x + w, y + h)


def detect_open_braces(gray, binv=None, exclude=()):
    """Diagonal thin OPEN-polyline braces -> boxes (x, y, w, h).

    Complements detect_braces, which only catches axis-aligned (tall/wide)
    braces. Diagonal braces have a near-square bbox, so they need a different
    signature: thin (low fill), whitespace-isolated, topologically open
    (no enclosed holes), and PRONGED -- sharp corners are what distinguish a
    grouping bracket from a straight leader line. Candidates whose bboxes
    mutually overlap are fragments of one part contour (concentric broken
    arcs) and are all dropped; ones overlapping an `exclude` box (braces the
    axis-aligned detector already found) are duplicates and skipped.
    """
    if binv is None:
        binv = binarize_inv(gray)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    cand = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < OPEN_MIN_AREA or area > OPEN_MAX_AREA:
            continue
        if max(w, h) < OPEN_MIN_LONG:
            continue
        if area / float(w * h) > OPEN_MAX_FILL:
            continue
        if ink_ratio_ring(binv, (x, y, w, h)) > OPEN_ISO_MAX_INK:
            continue
        if _hole_count(labels[y:y+h, x:x+w], i) > OPEN_MAX_HOLES:
            continue
        if _sharp_corners(labels, i, (x, y, w, h)) < OPEN_MIN_CORNERS:
            continue
        if any(iou(_xyxy((x, y, w, h)), _xyxy(e)) >= OPEN_MUTUAL_IOU
               for e in exclude):
            continue
        cand.append((x, y, w, h))
    # mutual-overlap suppression: fragments of one entity condemn each other
    out = [b for b in cand
           if not any(c is not b and iou(_xyxy(b), _xyxy(c)) >= OPEN_MUTUAL_IOU
                      for c in cand)]
    return out


def _sharp_vertices(mask):
    """approxPolyDP vertices with a sharp direction change (same parameters as
    _sharp_corners), as float (x, y) points in mask coordinates."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), 4, False)[:, 0, :]
    pts = pts.astype(np.float64)
    out = []
    for i in range(1, len(pts) - 1):
        v1, v2 = pts[i - 1] - pts[i], pts[i + 1] - pts[i]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 8 or n2 < 8:
            continue
        ang = np.degrees(np.arccos(np.clip(v1 @ v2 / (n1 * n2), -1, 1)))
        if ang < 140:
            out.append(pts[i])
    return out


def _notch_side(labels, idx, box, horizontal):
    """Which perpendicular side the brace's notch points to, or None.

    Catalogs draw the group id on the NOTCH side -- the mid-span cusp of a
    '{' and the apex of a V both point at the id; members align with the
    legs/span on the opposite side (ADR 0002, id-swap update). The notch is
    the deepest sharp vertex perpendicular from the pixel centroid, taken
    from the central 80% of the span only -- stroke end-hooks curl toward
    the member side and end-caps read as sharp vertices (the same trap the
    diagonal path documents). Returns (side_sign, depth_px) or None when no
    central sharp vertex exists (straight strokes, small 1-corner braces).
    """
    x, y, w, h = box
    mask = (labels[y:y + h, x:x + w] == idx).astype("uint8")
    span = w if horizontal else h
    a = 0 if horizontal else 1            # along-axis index in (x, y) verts
    p = 1 - a                             # perpendicular-axis index
    lo, hi = 0.1 * span, 0.9 * span
    central = [v for v in _sharp_vertices(mask) if lo <= v[a] <= hi]
    if not central:
        return None
    ys, xs = np.nonzero(mask)
    c = (xs.mean(), ys.mean())
    deep = max(central, key=lambda v: abs(v[p] - c[p]))
    off = deep[p] - c[p]
    return (1.0 if off > 0 else -1.0), abs(off)


def associate_open(open_braces, dets, binv, image_width, image_height):
    """Bind diagonal open-line braces to a group id + members.

    Measured geometry (sample.png + 4 catalog images, 12 true braces): the
    brace is a long diagonal SPINE spanning the grouped parts, with a short
    PRONG off the spine pointing at the group id; the id sits within ~1 digit
    height of the prong tip, member callouts hug the spine within its span on
    the opposite side. Axis-agnostic via PCA: the spine is the component's
    principal axis, sides are perpendicular projections. All tolerances are in
    digit heights -- the only scale stable across catalogs.

    Returns groups in associate()'s format. The id is required; members may be
    empty -- some braces group parts that carry no callout digits of their own
    (sample grp 5: washers/nuts identified only by the group id). A brace with
    no id near its prong tip stays unbound (JSON `open_braces` only).
    """
    # one labeling of the full image; each brace box IS a component's bbox
    # (detect_open_braces emits one box per component), so match it exactly --
    # picking "the largest component inside the box" can grab a part contour
    # that merely overlaps a big brace's bbox.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    by_bbox = {tuple(int(v) for v in stats[i][:4]): i for i in range(1, n)}

    groups = []
    for (bx, by, bw, bh) in open_braces:
        idx = by_bbox.get((bx, by, bw, bh))
        if idx is None:
            continue
        ys, xs = np.nonzero(labels[by:by + bh, bx:bx + bw] == idx)
        pts = np.column_stack([xs, ys]).astype(np.float64)
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
        u, v = vt[0], vt[1]                      # spine axis, perpendicular
        proj = (pts - c) @ u
        s0, s1 = proj.min() - OPEN_SPAN_PAD, proj.max() + OPEN_SPAN_PAD

        # prong tip: deepest sharp vertex off the spine, ends excluded -- the
        # caps of a thick stroke read as sharp vertices with a big excursion
        # (the documented PCA-endpoint trap), so only the central 80% of the
        # span may vote.
        lo, hi = s0 + 0.1 * (s1 - s0), s1 - 0.1 * (s1 - s0)
        sharp = [p for p in _sharp_vertices((labels[by:by + bh, bx:bx + bw] == idx).astype("uint8"))
                 if lo <= (p - c) @ u <= hi]
        if len(sharp) < OPEN_BIND_MIN_SHARP:
            continue
        tip = max(sharp, key=lambda p: abs((p - c) @ v))
        tip_side = 1.0 if (tip - c) @ v > 0 else -1.0
        tip_g = np.array([bx, by], float) + tip
        origin = np.array([bx, by], float) + c

        heights = sorted(d["bbox"][3] - d["bbox"][1] for d in dets)
        med_h = heights[len(heights) // 2] if heights else 0
        if not med_h:
            continue

        # the id: nearest callout to the prong tip, on the prong side, within
        # OPEN_ID_REACH of ITS OWN height (scale-free across catalogs)
        group, best = None, None
        in_span = []
        for d in dets:
            ctr = np.array([_cx(d), _cy(d)]) - origin
            along, off = ctr @ u, ctr @ v
            if not (s0 <= along <= s1):
                continue
            in_span.append((d, off))
            if off * tip_side > 0:
                dist = np.linalg.norm(np.array([_cx(d), _cy(d)]) - tip_g)
                if dist <= OPEN_ID_REACH * (d["bbox"][3] - d["bbox"][1]) \
                        and (best is None or dist < best):
                    group, best = d, dist
        if group is None:
            continue

        # members hug the spine on the opposite side from the id (may poke
        # slightly onto the id side); may be empty -- see docstring
        members = [d for (d, off) in in_span
                   if d is not group
                   and abs(off) <= OPEN_MEMBER_REACH * med_h
                   and off * tip_side <= OPEN_MEMBER_SIDE * med_h]
        groups.append({
            "group": group["text"],
            "group_bbox": group["bbox"],
            "brace_bbox": [bx, by, bx + bw, by + bh],
            "members": [{"text": d["text"], "bbox": d["bbox"]} for d in members],
        })
    return groups


def _cx(b):
    return (b["bbox"][0] + b["bbox"][2]) / 2.0


def _cy(b):
    return (b["bbox"][1] + b["bbox"][3]) / 2.0


def associate(braces, dets, image_width, image_height, binv=None):
    """Return groups: {group, group_bbox, brace_bbox, members:[...]}.

    Side selection: the brace's notch (cusp of '{', apex of V) points at the
    group id (ADR 0002 id-swap update) -- when `binv` is given and the brace
    component shows a notch deeper than NOTCH_MIN_DEPTH digit heights, the id
    pool is the notch side and members are the opposite side. Otherwise falls
    back to the count-vote (side with more aligned callouts = members), which
    is all there is for small 1-corner V-braces and when binv is unavailable.
    """
    labels = by_bbox = None
    if binv is not None:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(
            binv, connectivity=8)
        by_bbox = {tuple(int(v) for v in stats[i][:4]): i for i in range(1, n)}
    heights = sorted(d["bbox"][3] - d["bbox"][1] for d in dets)
    med_h = heights[len(heights) // 2] if heights else 0

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

        notch = None
        if labels is not None and med_h:
            idx = by_bbox.get((bx, by, bw, bh))
            if idx is not None:
                notch = _notch_side(labels, idx, (bx, by, bw, bh), horizontal)
                if notch and notch[1] < NOTCH_MIN_DEPTH * med_h:
                    notch = None

        pos_m = [d for o, d in pos if o <= member_reach]
        neg_m = [d for o, d in neg if o <= member_reach]
        if notch is not None:
            # id sits on the notch side; members on the other
            if notch[0] > 0:
                members, group_pool = neg_m, pos
            else:
                members, group_pool = pos_m, neg
        elif len(pos_m) >= len(neg_m):
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
