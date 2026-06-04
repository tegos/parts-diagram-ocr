"""Phase 1 spike (hybrid): connected-components find glyphs, EasyOCR recognizes crops.

Rationale: on clean line-art diagrams the ML text *detector* misses isolated
single digits, but classic CV finds every black blob trivially. So we detect
glyph candidates with connectedComponents, filter to digit-shaped blobs, group
neighbours into numbers, and use EasyOCR only to *read* each crop.

Usage:
    python -m src.spike_hybrid data/images/460127200.png [more.png ...]
Writes data/out/<name>_hybrid.png overlay + prints recognized numbers.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import easyocr

from src.config import OUT_DIR

# glyph filter heuristics (absolute px; diagrams are ~1700-2600 tall)
MIN_H, MAX_H = 16, 70          # single-digit glyph height
MIN_W, MAX_W = 6, 60           # single-digit glyph width
MIN_FILL, MAX_FILL = 0.18, 0.85  # area / bbox-area: rejects thin lines & solid blobs
MIN_AREA = 60

_reader = None


def reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def glyph_candidates(gray):
    """Return list of (x, y, w, h) blobs that look like single digit glyphs."""
    # black ink -> white foreground
    _, binv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(binv, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (MIN_H <= h <= MAX_H and MIN_W <= w <= MAX_W and area >= MIN_AREA):
            continue
        fill = area / float(w * h)
        if not (MIN_FILL <= fill <= MAX_FILL):
            continue
        out.append((x, y, w, h))
    return out


def group_numbers(boxes):
    """Merge horizontally-adjacent glyphs of similar height into number boxes."""
    boxes = sorted(boxes, key=lambda b: (b[1] // 20, b[0]))  # rough row order
    used = [False] * len(boxes)
    groups = []
    for i, b in enumerate(boxes):
        if used[i]:
            continue
        x, y, w, h = b
        used[i] = True
        cx2, cy = x + w, y + h // 2
        merged = True
        while merged:
            merged = False
            for j, c in enumerate(boxes):
                if used[j]:
                    continue
                jx, jy, jw, jh = c
                same_row = abs((jy + jh // 2) - cy) < 0.6 * h
                close = 0 <= (jx - cx2) < 0.8 * h
                similar = 0.6 < jh / h < 1.7
                if same_row and close and similar:
                    used[j] = True
                    x2 = max(cx2, jx + jw)
                    y = min(y, jy)
                    h = max(y + h, jy + jh) - y
                    cx2 = x2
                    merged = True
        groups.append((x, y, cx2 - x, h))
    return groups


def recognize(gray, box):
    x, y, w, h = box
    pad = 6
    crop = gray[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    res = reader().readtext(crop, allowlist="0123456789", detail=1,
                            text_threshold=0.4, low_text=0.3, mag_ratio=1.5)
    if not res:
        return None, 0.0
    _, text, conf = max(res, key=lambda r: r[2])
    return (text, conf) if text.isdigit() else (None, 0.0)


def run(path: Path):
    img = cv2.imread(str(path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cand = glyph_candidates(gray)
    groups = group_numbers(cand)

    dets = []
    for box in groups:
        text, conf = recognize(gray, box)
        if text is None or conf < 0.3:
            continue
        x, y, w, h = box
        dets.append({"text": text, "conf": round(float(conf), 3),
                     "bbox": [x, y, x + w, y + h]})
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, text, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}_hybrid.png"
    cv2.imwrite(str(out), img)
    print(f"\n{path.name}: {len(cand)} glyph cand -> {len(groups)} groups -> "
          f"{len(dets)} numbers -> {out}")
    for d in sorted(dets, key=lambda d: d["bbox"][1]):
        print(f"  {d['text']:>3}  conf={d['conf']:.2f}  bbox={d['bbox']}")
    return dets


def main(argv):
    if not argv:
        print("Give image path(s)", file=sys.stderr)
        return 1
    for a in argv:
        run(Path(a))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
