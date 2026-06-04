"""Unit tests for pure pipeline logic (no OCR / no image IO)."""
from src.glyphs import group_numbers, iou, nms
from src.pipeline import valid_number


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


def test_valid_number():
    assert valid_number("7")
    assert valid_number("29")
    assert not valid_number("0")
    assert not valid_number("012")   # leading zero
    assert not valid_number("100")   # out of range
    assert not valid_number("")
    assert not valid_number("1a")
