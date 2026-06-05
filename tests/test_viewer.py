"""Tests for the HTML viewer's testable core (`build_html`).

Uses a tiny synthetic PNG so no OCR/model is needed.
"""
import json
import re

import cv2
import numpy as np
import pytest

from src.viewer import build_html, main


@pytest.fixture
def tiny_png(tmp_path):
    img = np.full((100, 200, 3), 255, np.uint8)  # 200w x 100h white
    p = tmp_path / "tiny.png"
    cv2.imwrite(str(p), img)
    return p


def _data(html):
    """Pull the injected `const DATA = {...};` payload back out of the HTML."""
    m = re.search(r"const DATA = (\{.*?\});", html, re.S)
    return json.loads(m.group(1))


def _det(text, x0, y0, x1, y1):
    return {"text": text, "bbox": [x0, y0, x1, y1]}


def test_references_image_by_filename_not_base64(tiny_png):
    html = build_html(tiny_png, {"detections": [], "open_braces": []})
    assert 'src="tiny.png"' in html
    assert "base64" not in html


def test_img_src_override(tiny_png):
    html = build_html(tiny_png, {"detections": [], "open_braces": []},
                      img_src="images/tiny.png")
    assert 'src="images/tiny.png"' in html


def test_main_copies_image_next_to_html(tiny_png, tmp_path):
    out = tmp_path / "site"
    assert main([str(tiny_png), "--out", str(out)]) == 0
    assert (out / "tiny.html").exists()
    assert (out / "tiny.png").exists()
    assert 'src="tiny.png"' in (out / "tiny.html").read_text()


def test_one_box_per_detection_and_brace(tiny_png):
    result = {"detections": [_det("7", 0, 0, 10, 10), _det("8", 20, 0, 30, 10)],
              "open_braces": [[40, 40, 10, 20]]}
    data = _data(build_html(tiny_png, result))
    assert len(data["boxes"]) == 3
    assert sum(b["kind"] == "callout" for b in data["boxes"]) == 2
    assert sum(b["kind"] == "brace" for b in data["boxes"]) == 1


def test_header_reports_counts(tiny_png):
    result = {"detections": [_det("7", 0, 0, 10, 10)],
              "open_braces": [[40, 40, 10, 20], [60, 40, 10, 20]]}
    html = build_html(tiny_png, result)
    assert "1 callouts" in html and "2 braces" in html


def test_box_coords_are_percent_of_natural_size(tiny_png):
    # 200x100 image; a box at x0=20 -> 10%, y0=10 -> 10%, w=40 -> 20%, h=20 -> 20%
    result = {"detections": [_det("3", 20, 10, 60, 30)], "open_braces": []}
    box = _data(build_html(tiny_png, result))["boxes"][0]
    assert box["x"] == pytest.approx(10.0)
    assert box["y"] == pytest.approx(10.0)
    assert box["w"] == pytest.approx(20.0)
    assert box["h"] == pytest.approx(20.0)


def test_duplicate_numbers_share_siblings(tiny_png):
    # two "7"s should reference each other so they highlight together
    result = {"detections": [_det("7", 0, 0, 10, 10), _det("7", 50, 0, 60, 10),
                             _det("9", 80, 0, 90, 10)], "open_braces": []}
    data = _data(build_html(tiny_png, result))
    sevens = [b for b in data["boxes"] if b["text"] == "7"]
    assert len(sevens) == 2
    for b in sevens:
        assert set(b["siblings"]) == {sevens[0]["id"], sevens[1]["id"]}
    callout7 = next(c for c in data["callouts"] if c["text"] == "7")
    assert len(callout7["ids"]) == 2


def test_callouts_sorted_numerically(tiny_png):
    result = {"detections": [_det("10", 0, 0, 5, 5), _det("2", 10, 0, 15, 5),
                             _det("1A", 20, 0, 25, 5)], "open_braces": []}
    order = [c["text"] for c in _data(build_html(tiny_png, result))["callouts"]]
    assert order == ["1A", "2", "10"]


def test_missing_open_braces_key_is_ok(tiny_png):
    # _resolve may hand us {"detections": [...]} with no braces key
    data = _data(build_html(tiny_png, {"detections": [_det("5", 0, 0, 5, 5)]}))
    assert len(data["boxes"]) == 1
    assert data["braces"] == []


def _grp(label, group_bbox, brace_bbox, members):
    return {"group": label, "group_bbox": group_bbox,
            "brace_bbox": brace_bbox, "members": members}


def test_group_emits_label_and_brace_boxes(tiny_png):
    result = {"detections": [_det("7", 0, 0, 10, 10), _det("8", 20, 0, 30, 10)],
              "open_braces": [],
              "groups": [_grp("2", [50, 0, 60, 10], [0, 0, 100, 50],
                              [{"text": "7", "bbox": [0, 0, 10, 10]},
                               {"text": "8", "bbox": [20, 0, 30, 10]}])]}
    data = _data(build_html(tiny_png, result))
    kinds = [b["kind"] for b in data["boxes"]]
    assert kinds.count("group") == 1
    assert kinds.count("groupbrace") == 1
    g = data["groups"][0]
    det_ids = {b["id"] for b in data["boxes"] if b["kind"] == "callout"}
    assert set(g["members"]) == det_ids          # members resolved to detection ids
    assert g["label"] == "2"
    assert g["brace"] is not None
    label_box = next(b for b in data["boxes"] if b["kind"] == "group")
    assert label_box["id"] == g["id"]
    assert label_box["text"] == "grp 2"
    # label box siblings light label + brace together
    assert set(label_box["siblings"]) == {g["id"], g["brace"]}


def test_id_only_group_has_no_brace_box(tiny_png):
    result = {"detections": [_det("5", 0, 0, 10, 10)], "open_braces": [],
              "groups": [_grp("5", [0, 0, 10, 10], [0, 0, 50, 30], [])]}
    data = _data(build_html(tiny_png, result))
    g = data["groups"][0]
    assert g["brace"] is None
    assert g["members"] == []
    assert sum(b["kind"] == "groupbrace" for b in data["boxes"]) == 0


def test_unmatched_member_is_skipped(tiny_png):
    result = {"detections": [_det("7", 0, 0, 10, 10)], "open_braces": [],
              "groups": [_grp("3", [50, 0, 60, 10], [0, 0, 100, 50],
                              [{"text": "7", "bbox": [0, 0, 10, 10]},
                               {"text": "9", "bbox": [70, 0, 80, 10]}])]}
    g = _data(build_html(tiny_png, result))["groups"][0]
    assert len(g["members"]) == 1               # the "9" has no detection; skipped


def test_header_reports_group_count(tiny_png):
    result = {"detections": [], "open_braces": [],
              "groups": [_grp("1", [0, 0, 5, 5], [0, 0, 50, 50], [])]}
    assert "1 groups" in build_html(tiny_png, result)


def test_missing_groups_key_is_ok(tiny_png):
    data = _data(build_html(tiny_png, {"detections": [_det("5", 0, 0, 5, 5)]}))
    assert data["groups"] == []
