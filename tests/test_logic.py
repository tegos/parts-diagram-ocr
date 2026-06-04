"""Unit tests for pure pipeline logic (no OCR / no image IO)."""
import numpy as np
import cv2

from src.braces import associate, detect_open_braces
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


def test_open_brace_detects_diagonal_line():
    # a thin diagonal line: near-square bbox, so the aspect filter misses it,
    # but it is thin + isolated + open -> detect_open_braces catches it.
    g = _canvas()
    cv2.line(g, (30, 30), (170, 150), 0, 2)
    braces = detect_open_braces(g)
    assert len(braces) == 1


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
