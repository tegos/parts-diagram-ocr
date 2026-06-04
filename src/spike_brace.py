"""Phase 3 spike: find grouping braces as tall, thin connected components.

Grouping braces ('{') span several callout rows and group a range of part numbers.
They should be tall, thin, vertically-elongated blobs - distinct from both
digits (small) and the part drawing (large/bulky).

Usage:
    python -m src.spike_brace data/images/258103600.png
Writes data/out/<name>_brace.png (magenta = brace candidates, red = callouts).
"""
import json
import sys
from pathlib import Path

import cv2

from src.config import OUT_DIR

MIN_H = 60          # brace spans multiple rows
MIN_ASPECT = 3.0    # height / width
MAX_W = 80          # not the whole drawing
MAX_FILL = 0.35     # thin curve, not a solid block


def brace_candidates(gray):
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if h < MIN_H or w > MAX_W:
            continue
        if h / float(w) < MIN_ASPECT:
            continue
        if area / float(w * h) > MAX_FILL:
            continue
        out.append((x, y, w, h))
    return out


def run(path: Path):
    bgr = cv2.imread(str(path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    braces = brace_candidates(gray)

    jf = OUT_DIR / f"{path.stem}.json"
    dets = json.load(open(jf))["detections"] if jf.exists() else []

    for x, y, w, h in braces:
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (255, 0, 255), 2)
    for d in dets:
        b = d["bbox"]
        cv2.rectangle(bgr, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}_brace.png"
    cv2.imwrite(str(out), bgr)
    print(f"{path.name}: {len(braces)} brace candidates, {len(dets)} callouts "
          f"-> {out}")
    for x, y, w, h in sorted(braces, key=lambda b: b[1]):
        print(f"  brace x={x} y={y} w={w} h={h}")


def main(argv):
    if not argv:
        print("Give image path(s)", file=sys.stderr)
        return 1
    for a in argv:
        run(Path(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
