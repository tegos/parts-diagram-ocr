"""CLI: detect callout digits on parts diagrams.

Usage:
    python -m src.detect                      # all images in data/images
    python -m src.detect data/images/258103600.png ...
    python -m src.detect --no-overlay ...     # JSON only

Writes data/out/<name>.json (detections) and, unless --no-overlay,
data/out/<name>_overlay.png.
"""
import json
import sys
import time
from pathlib import Path

import cv2

from src.config import IMAGES_DIR, OUT_DIR
from src.pipeline import detect, draw_overlay


def collect(args):
    paths = [Path(a) for a in args]
    return paths if paths else sorted(IMAGES_DIR.glob("*.png"))


def process(path, write_overlay=True):
    bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"SKIP unreadable: {path}", file=sys.stderr)
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dets = detect(gray)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{path.stem}.json").write_text(
        json.dumps({"image": path.name, "detections": dets}, indent=2))
    if write_overlay:
        cv2.imwrite(str(OUT_DIR / f"{path.stem}_overlay.png"),
                    draw_overlay(bgr, dets))
    return dets


def main(argv):
    write_overlay = True
    if "--no-overlay" in argv:
        argv = [a for a in argv if a != "--no-overlay"]
        write_overlay = False

    paths = collect(argv)
    if not paths:
        print(f"No images in {IMAGES_DIR}", file=sys.stderr)
        return 1

    t0 = time.time()
    total = 0
    for p in paths:
        dets = process(p, write_overlay)
        if dets is None:
            continue
        total += len(dets)
        nums = ", ".join(d["text"] for d in sorted(dets, key=lambda d: d["bbox"][1]))
        print(f"{p.name}: {len(dets)} callouts  [{nums}]")

    dt = time.time() - t0
    print(f"\n{len(paths)} image(s), {total} callouts, {dt:.1f}s "
          f"({dt / max(1, len(paths)):.1f}s/img). JSON -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
