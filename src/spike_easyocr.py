"""Phase 1 spike: run EasyOCR on parts-diagram images, keep numeric callouts.

Usage:
    python -m src.spike_easyocr data/images/258103600.png [more.png ...]

For each image: prints detected numbers + confidence, writes an overlay PNG
to data/out/<name>_easyocr.png. Goal of the spike: eyeball recall on the
small isolated callout digits before committing to an engine.
"""
import sys
from pathlib import Path

import cv2
import easyocr

from src.config import OUT_DIR, UPSCALE, MIN_CONFIDENCE, DIGIT_RANGE

_reader = None


def reader():
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def in_range(text: str) -> bool:
    if not text.isdigit():
        return False
    return DIGIT_RANGE[0] <= int(text) <= DIGIT_RANGE[1]


def run(path: Path) -> list[dict]:
    img = cv2.imread(str(path))
    if img is None:
        print(f"SKIP unreadable: {path}", file=sys.stderr)
        return []

    big = cv2.resize(img, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC)
    raw = reader().readtext(big, allowlist="0123456789")

    dets = []
    for box, text, conf in raw:
        if conf < MIN_CONFIDENCE or not in_range(text):
            continue
        pts = [(int(x / UPSCALE), int(y / UPSCALE)) for x, y in box]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        dets.append({"text": text, "conf": round(float(conf), 3),
                     "bbox": [min(xs), min(ys), max(xs), max(ys)]})
        cv2.rectangle(img, (min(xs), min(ys)), (max(xs), max(ys)), (0, 0, 255), 2)
        cv2.putText(img, text, (min(xs), min(ys) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.stem}_easyocr.png"
    cv2.imwrite(str(out), img)
    print(f"\n{path.name}: {len(dets)} numeric callouts -> {out}")
    for d in sorted(dets, key=lambda d: -d["conf"]):
        print(f"  {d['text']:>3}  conf={d['conf']:.3f}  bbox={d['bbox']}")
    return dets


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv]
    if not paths:
        print("Give at least one image path", file=sys.stderr)
        return 1
    for p in paths:
        run(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
