"""Dump gate metrics for every BOUND brace (axis and diagonal), true vs false.

Re-runs the geometry on cached detections (data/out/<stem>.json must exist),
so no OCR pass is needed. For each bound group prints the metrics a gate
could cut on; cross-check ids against eval/groundtruth to mark true/false.

Usage: .venv/bin/python scripts/measure_braces.py <stem> [<stem>...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.braces import (OPEN_SPAN_PAD, _cx, _cy, _notch_side, _sharp_corners,
                        _sharp_vertices, associate, detect_braces,
                        detect_open_braces)
from src.glyphs import binarize_inv


def main(stems):
    for stem in stems:
        gt_file = Path("eval/groundtruth") / f"{stem}.json"
        gt_groups = (json.loads(gt_file.read_text()).get("groups", {})
                     if gt_file.exists() else {})
        dets = json.loads((Path("data/out") / f"{stem}.json").read_text())["detections"]
        gray = cv2.cvtColor(cv2.imread(f"data/images/{stem}.png"),
                            cv2.COLOR_BGR2GRAY)
        binv = binarize_inv(gray)
        H, W = gray.shape
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
        by_bbox = {tuple(int(v) for v in stats[i][:4]): i for i in range(1, n)}
        heights = sorted(d["bbox"][3] - d["bbox"][1] for d in dets)
        med_h = heights[len(heights) // 2] if heights else 0

        # ---- axis-aligned path: per bound brace, corner + notch metrics ----
        closed = detect_braces(gray, binv)
        for g in associate(closed, dets, W, H, binv=binv):
            bx0, by0, bx1, by1 = g["brace_bbox"]
            box = (bx0, by0, bx1 - bx0, by1 - by0)
            idx = by_bbox.get(box)
            corners = _sharp_corners(labels, idx, box) if idx is not None else -1
            area = int(stats[idx][4]) if idx is not None else -1
            fill = area / float(box[2] * box[3]) if idx is not None else -1
            tag = "TRUE " if g["group"] in gt_groups else "FALSE"

            horizontal = box[2] >= box[3]
            if horizontal:
                span0, span1 = bx0, bx1
                along, perp, perp_c = _cx, _cy, (by0 + by1) / 2.0
            else:
                span0, span1 = by0, by1
                along, perp, perp_c = _cy, _cx, (bx0 + bx1) / 2.0
            notch = (_notch_side(labels, idx, box, horizontal)
                     if idx is not None else None)
            n_side = "+" if notch and notch[0] > 0 else "-" if notch else "?"
            n_depth = notch[1] / med_h if notch and med_h else -1
            cur_side = ("+" if perp({"bbox": g["group_bbox"]}) - perp_c > 0
                        else "-")
            in_span = " ".join(
                f"{d['text']}{'+' if perp(d) - perp_c > 0 else '-'}"
                for d in dets if span0 <= along(d) <= span1)
            print(f"{stem:<12} AXIS {tag} id={g['group']:<4} "
                  f"corners={corners:3d} long={max(box[2], box[3]):4d} "
                  f"area={area:5d} fill={fill:.3f} "
                  f"members={len(g['members'])} "
                  f"notch={n_side}{n_depth:5.2f} cur_id_side={cur_side} "
                  f"in_span=[{in_span}]")

        # ---- diagonal path: spine/prong/id-dist in digit heights ----
        obs = detect_open_braces(gray, binv, exclude=closed)
        for (bx, by, bw, bh) in obs:
            idx = by_bbox.get((bx, by, bw, bh))
            if idx is None or not med_h:
                continue
            ys, xs = np.nonzero(labels[by:by + bh, bx:bx + bw] == idx)
            pts = np.column_stack([xs, ys]).astype(np.float64)
            c = pts.mean(0)
            _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
            u, v = vt[0], vt[1]
            proj = (pts - c) @ u
            s0, s1 = proj.min() - OPEN_SPAN_PAD, proj.max() + OPEN_SPAN_PAD
            lo, hi = s0 + 0.1 * (s1 - s0), s1 - 0.1 * (s1 - s0)
            mask = (labels[by:by + bh, bx:bx + bw] == idx).astype("uint8")
            sharp = [p for p in _sharp_vertices(mask) if lo <= (p - c) @ u <= hi]
            if not sharp:
                continue
            tip = max(sharp, key=lambda p: abs((p - c) @ v))
            tip_side = 1.0 if (tip - c) @ v > 0 else -1.0
            tip_g = np.array([bx, by], float) + tip
            origin = np.array([bx, by], float) + c
            group, best, in_span = None, None, []
            for d in dets:
                ctr = np.array([_cx(d), _cy(d)]) - origin
                along, off = ctr @ u, ctr @ v
                if not (s0 <= along <= s1):
                    continue
                in_span.append((d, off))
                if off * tip_side > 0:
                    dist = np.linalg.norm(np.array([_cx(d), _cy(d)]) - tip_g)
                    ih = d["bbox"][3] - d["bbox"][1]
                    if dist <= 1.3 * ih and (best is None or dist < best):
                        group, best = d, dist
            if group is None:
                continue
            gh = group["bbox"][3] - group["bbox"][1]
            members = [d for (d, off) in in_span
                       if d is not group and abs(off) <= 4.5 * med_h
                       and off * tip_side <= 1.0 * med_h]
            area = int(stats[idx][4])
            tag = "TRUE " if group["text"] in gt_groups else "FALSE"
            print(f"{stem:<12} DIAG {tag} id={group['text']:<4} "
                  f"spine={(s1 - s0) / med_h:5.1f} "
                  f"prong={abs((tip - c) @ v) / med_h:4.1f} "
                  f"id_dist={best / gh:4.2f} members={len(members):2d} "
                  f"area={area:5d} fill={area / float(bw * bh):.3f} "
                  f"corners={len(sharp):2d}")


if __name__ == "__main__":
    main(sys.argv[1:])
