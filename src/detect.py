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

from src.braces import associate, detect_braces, detect_open_braces
from src.glyphs import binarize_inv
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
    binv = binarize_inv(gray)
    dets = detect(gray)
    braces = detect_braces(gray, binv)
    groups = associate(braces, dets, gray.shape[1], gray.shape[0])
    open_braces = detect_open_braces(gray, binv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{path.stem}.json").write_text(json.dumps(
        {"image": path.name, "detections": dets, "groups": groups,
         "open_braces": [list(b) for b in open_braces]}, indent=2))
    if write_overlay:
        cv2.imwrite(str(OUT_DIR / f"{path.stem}_overlay.png"),
                    draw_overlay(bgr, dets, groups, open_braces))
    return dets, groups


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
    total_groups = 0
    for p in paths:
        res = process(p, write_overlay)
        if res is None:
            continue
        dets, groups = res
        total += len(dets)
        total_groups += len(groups)
        nums = ", ".join(d["text"] for d in sorted(dets, key=lambda d: d["bbox"][1]))
        print(f"{p.name}: {len(dets)} callouts, {len(groups)} groups  [{nums}]")

    dt = time.time() - t0
    print(f"\n{len(paths)} image(s), {total} callouts, {total_groups} groups, "
          f"{dt:.1f}s ({dt / max(1, len(paths)):.1f}s/img). JSON -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
