"""Phase 3 spike: detect leader lines and test proximity to detected callouts.

Goal: see whether HoughLinesP isolates the thin leader lines that connect a
callout number to its part, so we can (a) associate digit -> part and (b) drop
interior false positives that have no leader.

Usage:
    python -m src.spike_leader data/images/258103600.png
Writes data/out/<name>_leader.png (green = lines, red = callouts, yellow =
callouts WITH a nearby line endpoint).
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from src.config import OUT_DIR

MIN_LEN = 40        # leader lines are long
MAX_GAP = 8
NEAR = 35           # px: line endpoint within this of a digit bbox -> linked


def detect_lines(gray):
    edges = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50,
                            minLineLength=MIN_LEN, maxLineGap=MAX_GAP)
    return [] if lines is None else [tuple(l[0]) for l in lines]


def near_box(pt, box, r):
    x0, y0, x1, y1 = box
    px, py = pt
    cx = min(max(px, x0), x1)
    cy = min(max(py, y0), y1)
    return (px - cx) ** 2 + (py - cy) ** 2 <= r * r


def run(path: Path):
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    lines = detect_lines(gray)

    jf = OUT_DIR / f"{path.stem}.json"
    dets = json.load(open(jf))["detections"] if jf.exists() else []

    for x1, y1, x2, y2 in lines:
        cv2.line(bgr, (x1, y1), (x2, y2), (0, 180, 0), 1)

    linked = 0
    for d in dets:
        box = d["bbox"]
        has = any(near_box((x1, y1), box, NEAR) or near_box((x2, y2), box, NEAR)
                  for x1, y1, x2, y2 in lines)
        linked += has
        color = (0, 215, 255) if has else (0, 0, 255)
        cv2.rectangle(bgr, (box[0], box[1]), (box[2], box[3]), color, 2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}_leader.png"
    cv2.imwrite(str(out), bgr)
    print(f"{path.name}: {len(lines)} lines, {len(dets)} callouts, "
          f"{linked} linked, {len(dets) - linked} unlinked -> {out}")


def main(argv):
    if not argv:
        print("Give image path(s)", file=sys.stderr)
        return 1
    for a in argv:
        run(Path(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
