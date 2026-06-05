"""Unit tests for pure pipeline logic (no OCR / no image IO)."""
import numpy as np
import cv2

from src.braces import associate, associate_open, detect_open_braces, _notch_side
from src.glyphs import glyph_candidates, group_numbers, iou, nms
from src.pipeline import valid_callout


def _det(text, cx, cy):
    return {"text": text, "conf": 1.0, "bbox": [cx - 5, cy - 5, cx + 5, cy + 5]}


def test_associate_left_brace_opens_right():
    # left-margin brace at x=100 spanning y=100..400; group-id "15" to its left,
    # members 17/18 to its right within span; "99" is outside the span.
    braces = [(100, 100, 20, 300)]
    dets = [_det("15", 60, 250), _det("17", 200, 150),
            _det("18", 200, 350), _det("99", 200, 900)]
    groups = associate(braces, dets, image_width=1000, image_height=1000)
    assert len(groups) == 1
    g = groups[0]
    assert g["group"] == "15"
    assert sorted(m["text"] for m in g["members"]) == ["17", "18"]


def test_associate_right_brace_opens_left():
    braces = [(900, 100, 20, 300)]
    dets = [_det("16", 950, 250), _det("20", 800, 150), _det("21", 800, 350)]
    groups = associate(braces, dets, image_width=1000, image_height=1000)
    assert groups[0]["group"] == "16"
    assert sorted(m["text"] for m in groups[0]["members"]) == ["20", "21"]


def test_associate_horizontal_brace():
    # wide brace spanning x=200..500 at y=300; group-id "21" above, members below
    braces = [(200, 300, 300, 20)]
    dets = [_det("21", 350, 250), _det("6", 240, 380),
            _det("7", 350, 380), _det("5", 460, 380)]
    groups = associate(braces, dets, image_width=1000, image_height=1000)
    assert groups[0]["group"] == "21"
    assert sorted(m["text"] for m in groups[0]["members"]) == ["5", "6", "7"]


def _canvas():
    return np.full((220, 220), 255, np.uint8)


def test_open_brace_detects_pronged_polyline():
    # a grouping brace drawn as a diagonal polyline WITH prongs (direction
    # changes) -- the prongs are what make it a brace and not a leader line.
    g = _canvas()
    pts = np.array([[20, 60], [50, 40], [90, 70], [120, 50], [160, 80],
                    [170, 120], [140, 150], [150, 190]], np.int32)
    cv2.polylines(g, [pts], False, 0, 2)
    braces = detect_open_braces(g)
    assert len(braces) == 1


def test_open_brace_rejects_straight_line():
    # a plain straight diagonal line is a LEADER line (number -> part), not a
    # brace: no prongs means no grouping semantics. Used to be accepted.
    g = _canvas()
    cv2.line(g, (30, 30), (170, 150), 0, 2)
    assert detect_open_braces(g) == []


def test_open_brace_rejects_concentric_fragments():
    # two broken arcs of the same circle (a part contour, e.g. a flywheel rim
    # interrupted by teeth) overlap each other's bboxes -> both dropped.
    g = np.full((400, 400), 255, np.uint8)
    for r, gap in ((150, 30), (120, 60)):
        for a0 in range(0, 360, 90):
            cv2.ellipse(g, (200, 200), (r, r), 0, a0 + gap / 4, a0 + 90 - gap / 4, 0, 2)
    # make each arc prong-y enough to pass the corner gate on its own
    assert detect_open_braces(g) == []


def test_open_brace_excludes_axis_brace_duplicates():
    # a candidate overlapping an already-detected axis-aligned brace box is the
    # SAME brace caught twice -> the open detector must not re-report it.
    g = _canvas()
    pts = np.array([[20, 60], [50, 40], [90, 70], [120, 50], [160, 80],
                    [170, 120], [140, 150], [150, 190]], np.int32)
    cv2.polylines(g, [pts], False, 0, 2)
    found = detect_open_braces(g)
    assert len(found) == 1
    assert detect_open_braces(g, exclude=found) == []


def test_open_brace_rejects_closed_loop():
    # a thin large rectangle OUTLINE encloses a hole -> rejected (part contour).
    g = _canvas()
    cv2.rectangle(g, (30, 30), (190, 190), 0, 1)
    assert detect_open_braces(g) == []


def test_open_brace_rejects_solid_blob():
    # a filled block is not thin (high fill) -> rejected.
    g = _canvas()
    cv2.rectangle(g, (60, 60), (150, 150), 0, -1)
    assert detect_open_braces(g) == []


def _checker(rows, cols):
    """A single 8-connected blob filling `rows`x`cols` at ~50% (checkerboard)."""
    binv = np.zeros((rows + 8, cols + 8), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i + j) % 2 == 0:
                binv[2 + i, 2 + j] = 255
    return binv


def test_glyph_accepts_narrow_one():
    # a narrow digit "1" (w5, h17, ~half fill, aspect 3.4) must pass -- this is
    # exactly the shape GLYPH_MIN_W=6 used to drop.
    cands = glyph_candidates(_checker(17, 5))
    assert len(cands) == 1
    x, y, w, h = cands[0]
    assert w == 5 and h == 17


def test_glyph_rejects_tilted_bar():
    # a TILTED part-edge segment (h ~2x the digit size, w inflated by the tilt
    # so the aspect cap misses it) reads as "1" at conf 1.0 — must die at the
    # candidate stage. A bare bar fits a straight line near-perfectly
    # (resid/len < 0.03); a real "1" has a serif/flag (0.07+).
    g = np.full((80, 30), 255, np.uint8)
    cv2.line(g, (8, 5), (20, 65), 0, 2)
    from src.glyphs import binarize_inv
    assert glyph_candidates(binarize_inv(g)) == []


def test_glyph_rejects_tilted_bars_on_194103800():
    # regression guard: three valve-cover edge segments at (506,2136),
    # (940,1887), (852,1935) sat in whitespace pockets and read as "1"
    # conf 0.97-1.0. All real digits on that diagram are h 29-32; the bars
    # are h 56-62 bare lines.
    from src.config import IMAGES_DIR
    p = IMAGES_DIR / "194103800.png"
    if not p.exists():
        import pytest
        pytest.skip("194103800.png not present")
    from src.glyphs import binarize_inv
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    cands = glyph_candidates(binarize_inv(g))
    for (bx, by) in [(506, 2136), (940, 1887), (852, 1935)]:
        assert not any(abs(x - bx) < 4 and abs(y - by) < 4 for (x, y, w, h) in cands), \
            f"bare bar at ({bx},{by}) still a candidate"
    # the real "1" at (705,365) must survive (serif/flag -> not a bare line)
    assert any(abs(x - 705) < 4 and abs(y - 365) < 4 for (x, y, w, h) in cands)


def test_glyph_rejects_tall_thin_bar():
    # a leader-line fragment (w4, h36, aspect 9) reads as "1" to OCR but must be
    # rejected at the candidate stage by the aspect cap.
    assert glyph_candidates(_checker(36, 4)) == []


def test_glyph_finds_narrow_one_on_sample():
    # regression guard: the group-1 id "1" (a w4 digit at ~x552,y624) is a real
    # callout the old width gate dropped; it must now be a candidate.
    from src.config import DATA_DIR
    p = DATA_DIR / "sample.png"
    if not p.exists():
        import pytest
        pytest.skip("sample.png not present")
    from src.glyphs import binarize_inv
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    cands = glyph_candidates(binarize_inv(g))
    hit = [b for b in cands if b[0] <= 554 <= b[0] + b[2] and b[1] <= 632 <= b[1] + b[3]]
    assert hit, "narrow '1' near (552,624) not recovered"


def test_open_brace_finds_three_on_sample():
    # regression guard: the alternator sample has exactly three diagonal grouping
    # braces (over parts 1, 2, 5) that the axis-aligned detector misses entirely.
    from src.config import DATA_DIR
    p = DATA_DIR / "sample.png"
    if not p.exists():
        import pytest
        pytest.skip("sample.png not present")
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert len(detect_open_braces(g)) == 3


def _pronged_scene():
    """A diagonal spine pronged along its length, on a 400x400 canvas.

    Geometry mirrors the measured sample.png braces: a real grouping bracket
    is pronged THROUGHOUT its spine, carrying 6-9 sharp vertices in the
    central 80% of the span (sample.png braces: 6/6/10); a part-contour
    fragment shows only ~3. Members hug the spine (offset ~5px), the id sits
    past the DEEPEST prong tip (offset ~35px, beyond the member band).
    """
    g = np.full((400, 400), 255, np.uint8)
    cv2.line(g, (40, 200), (360, 120), 0, 2)     # spine
    cv2.line(g, (120, 175), (123, 187), 0, 2)    # short prong tick
    cv2.line(g, (200, 160), (205, 185), 0, 2)    # deepest prong toward the id
    cv2.line(g, (280, 142), (283, 154), 0, 2)    # short prong tick
    from src.glyphs import binarize_inv
    binv = binarize_inv(g)
    # the box must be the component's exact bbox, as detect_open_braces emits
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    i = max(range(1, n), key=lambda k: stats[k][4])
    box = tuple(int(v) for v in stats[i][:4])
    return binv, box


def test_associate_open_binds_id_and_members():
    binv, box = _pronged_scene()
    dets = [_det("5", 211, 191),     # id: within ~1 digit height of the prong tip
            _det("3", 120, 185),     # member: hugs the spine
            _det("4", 280, 142),     # member: hugs the spine
            _det("9", 60, 350)]      # far away: ignored
    groups = associate_open([box], dets, binv, 400, 400)
    assert len(groups) == 1
    g = groups[0]
    assert g["group"] == "5"
    assert sorted(m["text"] for m in g["members"]) == ["3", "4"]


def test_associate_open_unbound_without_id():
    # no callout near the prong tip -> no group (stays JSON-only)
    binv, box = _pronged_scene()
    dets = [_det("3", 120, 185), _det("4", 280, 142)]
    assert associate_open([box], dets, binv, 400, 400) == []


def test_associate_open_binds_id_only():
    # id at the prong but no callouts on the spine: still a group -- some
    # braces group parts that carry no callout digits (sample grp 5).
    binv, box = _pronged_scene()
    dets = [_det("5", 211, 191), _det("9", 60, 350)]
    groups = associate_open([box], dets, binv, 400, 400)
    assert len(groups) == 1
    assert groups[0]["group"] == "5"
    assert groups[0]["members"] == []


def test_associate_open_rejects_sparse_corner_fragment():
    # a part-contour fragment shows at most ~3 sharp vertices along its
    # spine; a real grouping bracket is pronged throughout (measured: true
    # braces 6-9 central sharp vertices, the two dataset false bindings 3).
    # Same id/member layout as the binding tests -- only the prong count
    # differs, so this isolates the OPEN_BIND_MIN_SHARP gate.
    g = np.full((400, 400), 255, np.uint8)
    cv2.line(g, (40, 200), (360, 120), 0, 2)     # spine
    cv2.line(g, (200, 160), (205, 185), 0, 2)    # lone prong tick
    from src.glyphs import binarize_inv
    binv = binarize_inv(g)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    i = max(range(1, n), key=lambda k: stats[k][4])
    box = tuple(int(v) for v in stats[i][:4])
    dets = [_det("5", 211, 191),     # id: within ~1 digit height of the prong tip
            _det("3", 120, 185),     # member: hugs the spine
            _det("4", 280, 142),     # member: hugs the spine
            _det("9", 60, 350)]      # far away: ignored
    assert associate_open([box], dets, binv, 400, 400) == []


def test_associate_open_binds_sample_braces():
    # regression: sample.png has 3 diagonal-brace groups (ids 2, 1, 5).
    # grp 2 -> members 7,8; grp 1 -> 3,4 (+ nested 5,6 -- known double-count);
    # grp 5 binds id-only (its washers/nuts carry no callout digits).
    from src.config import DATA_DIR
    p = DATA_DIR / "sample.png"
    if not p.exists():
        import pytest
        pytest.skip("sample.png not present")
    import json
    from src.glyphs import binarize_inv
    from src.braces import detect_braces, detect_open_braces
    g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    binv = binarize_inv(g)
    result = json.loads((DATA_DIR / "out" / "sample.json").read_text())
    dets = result["detections"]
    ob = detect_open_braces(g, binv, exclude=detect_braces(g, binv))
    groups = associate_open(ob, dets, binv, g.shape[1], g.shape[0])
    by_id = {grp["group"]: sorted(m["text"] for m in grp["members"])
             for grp in groups}
    assert "2" in by_id and by_id["2"] == ["7", "8"]
    assert "1" in by_id and {"3", "4"} <= set(by_id["1"])
    assert "5" in by_id    # id-only: bound with no members


def _component_box(binv):
    """bbox (x, y, w, h) of the single foreground component in a test canvas."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    assert n == 2, f"expected exactly 1 component, got {n - 1}"
    return labels, tuple(int(v) for v in stats[1][:4])


def test_notch_side_v_brace_points_down():
    # horizontal V-brace, apex pointing DOWN (id below, members above) --
    # the 627253920 / 194103550 layout
    binv = np.zeros((200, 400), np.uint8)
    pts = np.array([[50, 60], [200, 140], [350, 60]], np.int32)
    cv2.polylines(binv, [pts], False, 255, 3)
    labels, box = _component_box(binv)
    side, depth = _notch_side(labels, 1, box, horizontal=True)
    assert side > 0          # apex below the centroid (y grows down)
    assert depth > 20


def test_notch_side_curly_cusp_left():
    # vertical '{' opening right: cusp pokes LEFT at mid-height (id left,
    # members right) -- the 258103600 layout
    binv = np.zeros((400, 200), np.uint8)
    pts = np.array([[110, 50], [100, 80], [100, 180], [60, 200],
                    [100, 220], [100, 320], [110, 350]], np.int32)
    cv2.polylines(binv, [pts], False, 255, 3)
    labels, box = _component_box(binv)
    side, depth = _notch_side(labels, 1, box, horizontal=False)
    assert side < 0          # cusp left of the centroid
    assert depth > 20


def test_notch_side_straight_line_returns_none():
    # a plain vertical stroke has no central sharp vertex -> no notch signal
    binv = np.zeros((400, 200), np.uint8)
    cv2.line(binv, (100, 50), (100, 350), 255, 3)
    labels, box = _component_box(binv)
    assert _notch_side(labels, 1, box, horizontal=False) is None


def test_associate_skips_brace_with_no_members():
    braces = [(100, 100, 20, 300)]
    dets = [_det("15", 60, 250)]  # only an outer callout, no inner members
    assert associate(braces, dets, image_width=1000, image_height=1000) == []


def test_associate_skips_brace_with_no_group_id():
    # members present but nothing on the outer side -> no group id -> dropped
    braces = [(100, 100, 20, 300)]
    dets = [_det("17", 200, 150), _det("18", 200, 350)]
    assert associate(braces, dets, image_width=1000, image_height=1000) == []


def test_group_merges_adjacent_same_row():
    # two glyphs side by side, height 30, gap 10 -> one number
    boxes = [(100, 100, 20, 30), (125, 100, 20, 30)]
    groups = group_numbers(boxes)
    assert len(groups) == 1
    x, y, w, h = groups[0]
    assert x == 100 and w == 45


def test_group_merges_across_y_bin_boundary():
    # regression (710422200 "14"): two same-row glyphs whose y values fall in
    # different 20px sort bins (2239 vs 2240) were visited out of x order, and
    # the right glyph can't merge leftward -> "14" split into "1" + "4".
    boxes = [(1331, 2240, 10, 37), (1354, 2239, 26, 38)]
    groups = group_numbers(boxes)
    assert len(groups) == 1
    x, y, w, h = groups[0]
    assert x == 1331 and x + w == 1380


def test_group_keeps_distant_apart():
    boxes = [(100, 100, 20, 30), (400, 100, 20, 30)]
    assert len(group_numbers(boxes)) == 2


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_nms_drops_overlap_keeps_best():
    dets = [
        {"text": "5", "conf": 0.9, "bbox": [0, 0, 10, 10]},
        {"text": "5", "conf": 0.6, "bbox": [1, 1, 11, 11]},
        {"text": "7", "conf": 0.8, "bbox": [50, 50, 60, 60]},
    ]
    kept = nms(dets, 0.5)
    assert len(kept) == 2
    assert {d["conf"] for d in kept} == {0.9, 0.8}


def test_recognize_split_concatenates_isolated_glyphs(monkeypatch):
    # whole-box OCR misreads a digit beside a suffix (open-top "4" + "A" -> "AA");
    # the split fallback reads each connected component in isolation, where each
    # is unambiguous, and concatenates left-to-right.
    import src.ocr as ocr
    binv = np.zeros((50, 100), np.uint8)
    binv[10:40, 10:35] = 255   # left glyph (the "4")
    binv[10:40, 55:80] = 255   # right glyph (the "A")
    monkeypatch.setattr(ocr, "recognize",
                        lambda gray, box: ("4", 1.0) if box[0] < 45 else ("A", 0.9))
    text, conf = ocr.recognize_split(None, binv, (0, 0, 100, 50))
    assert text == "4A"
    assert conf == 0.9          # group confidence is the weakest glyph


def test_recognize_split_skips_single_glyph(monkeypatch):
    # one blob -> splitting can't help (whole == part); signal no result so the
    # caller keeps the original (dropped) read instead of looping.
    import src.ocr as ocr
    binv = np.zeros((50, 100), np.uint8)
    binv[10:40, 10:35] = 255
    monkeypatch.setattr(ocr, "recognize", lambda gray, box: ("4", 1.0))
    assert ocr.recognize_split(None, binv, (0, 0, 100, 50)) == (None, 0.0)


def test_recognize_split_rejects_too_many_glyphs(monkeypatch):
    # 5 blobs is not a callout (max is 3 digits + 1 suffix) -> no result.
    import src.ocr as ocr
    binv = np.zeros((50, 140), np.uint8)
    for i in range(5):
        binv[10:40, 10 + i * 25:30 + i * 25] = 255
    monkeypatch.setattr(ocr, "recognize", lambda gray, box: ("1", 1.0))
    assert ocr.recognize_split(None, binv, (0, 0, 140, 50)) == (None, 0.0)


def test_recognize_split_rejects_multichar_component(monkeypatch):
    # a single narrow blob can read as "11" in isolation; trusting it would turn
    # a 2-blob "12" box into a bogus "112". A component that reads >1 char means
    # the split is unreliable -> no result, keep the original read.
    import src.ocr as ocr
    binv = np.zeros((50, 100), np.uint8)
    binv[10:40, 10:35] = 255
    binv[10:40, 55:80] = 255
    monkeypatch.setattr(ocr, "recognize",
                        lambda gray, box: ("11", 0.9) if box[0] < 45 else ("2", 0.9))
    assert ocr.recognize_split(None, binv, (0, 0, 100, 50)) == (None, 0.0)


def test_valid_callout():
    assert valid_callout("7")
    assert valid_callout("29")
    assert valid_callout("116")        # catalogs exceed 100
    assert valid_callout("1A")         # suffix callouts
    assert valid_callout("16B")
    assert not valid_callout("0")
    assert not valid_callout("012")   # leading zero
    assert not valid_callout("250")   # out of range
    assert not valid_callout("")
    assert not valid_callout("A")     # bare letter
    assert not valid_callout("1C")    # unsupported suffix


def _curly_brace_binv():
    """1000x1000 canvas with one '{' opening right; cusp points LEFT at
    mid-span (y=250). Returns (binv, brace box exactly matching the
    component bbox -- associate matches components by exact bbox)."""
    binv = np.zeros((1000, 1000), np.uint8)
    pts = np.array([[210, 100], [200, 130], [200, 230], [160, 250],
                    [200, 270], [200, 370], [210, 400]], np.int32)
    cv2.polylines(binv, [pts], False, 255, 3)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    assert n == 2
    return binv, tuple(int(v) for v in stats[1][:4])


def test_associate_notch_fixes_id_swap():
    # the 258103600 pattern: id (24) on the cusp side plus a stray callout
    # (27) on the same side -- count-vote sees 2 left vs 1 right and makes
    # the LEFT side the member side, emitting 25{24,27}. The cusp points
    # left: with binv the id pool is the left side and 24 wins at the tip.
    binv, box = _curly_brace_binv()
    dets = [_det("24", 100, 250),    # drawn id, at the cusp
            _det("27", 100, 390),    # stray same-side callout near span end
            _det("25", 270, 250)]    # the drawn member (within GROUP_REACH=100
                                     # of the spine, so the legacy path really
                                     # does pick it as the id -- see next test)
    groups = associate([box], dets, image_width=1000, image_height=1000,
                       binv=binv)
    assert len(groups) == 1
    assert groups[0]["group"] == "24"
    assert [m["text"] for m in groups[0]["members"]] == ["25"]


def test_associate_without_binv_keeps_count_vote():
    # no binv -> no notch signal -> legacy behavior (documents the swap)
    dets = [_det("24", 100, 250), _det("27", 100, 390), _det("25", 270, 250)]
    box = _curly_brace_binv()[1]
    groups = associate([box], dets, image_width=1000, image_height=1000)
    assert groups[0]["group"] == "25"
    assert sorted(m["text"] for m in groups[0]["members"]) == ["24", "27"]
