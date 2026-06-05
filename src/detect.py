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
from contextlib import contextmanager
from pathlib import Path

import cv2

from src.braces import (associate, associate_open, detect_braces,
                        detect_open_braces)
from src.glyphs import binarize_inv
from src.config import IMAGES_DIR, OUT_DIR
from src.pipeline import detect, draw_overlay


class StageTimer:
    """Accumulates wall time per pipeline stage across a batch run."""
    def __init__(self):
        self.acc = {}

    @contextmanager
    def stage(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.acc[name] = self.acc.get(name, 0.0) + time.perf_counter() - t0

    def report(self):
        total = sum(self.acc.values()) or 1.0
        for name, s in sorted(self.acc.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<16}{s:8.1f}s  {s / total:5.0%}")


def collect(args):
    paths = [Path(a) for a in args]
    return paths if paths else sorted(IMAGES_DIR.glob("*.png"))


def process(path, write_overlay=True, timer=None):
    tm = timer or StageTimer()
    with tm.stage("imread"):
        bgr = cv2.imread(str(path))
    if bgr is None:
        print(f"SKIP unreadable: {path}", file=sys.stderr)
        return None
    with tm.stage("binarize"):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        binv = binarize_inv(gray)
    with tm.stage("ocr_detect"):
        dets = detect(gray)
    with tm.stage("braces_closed"):
        braces = detect_braces(gray, binv)
        groups = associate(braces, dets, gray.shape[1], gray.shape[0])
    with tm.stage("braces_open"):
        open_braces = detect_open_braces(gray, binv, exclude=braces)
        groups += associate_open(open_braces, dets, binv,
                                 gray.shape[1], gray.shape[0])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tm.stage("json_write"):
        (OUT_DIR / f"{path.stem}.json").write_text(json.dumps(
            {"image": path.name, "detections": dets, "groups": groups,
             "open_braces": [list(b) for b in open_braces]}, indent=2))
    if write_overlay:
        with tm.stage("overlay_write"):
            cv2.imwrite(str(OUT_DIR / f"{path.stem}_overlay.png"),
                        draw_overlay(bgr, dets, groups))
    return dets, groups


def main(argv):
    write_overlay = True
    if "--no-overlay" in argv:
        argv = [a for a in argv if a != "--no-overlay"]
        write_overlay = False

    profile = False
    if "--profile" in argv:
        argv = [a for a in argv if a != "--profile"]
        profile = True

    paths = collect(argv)
    if not paths:
        print(f"No images in {IMAGES_DIR}", file=sys.stderr)
        return 1

    t0 = time.time()
    total = 0
    total_groups = 0
    timer = StageTimer() if profile else None
    first_dt = None
    for i, p in enumerate(paths):
        if profile and i == 0:
            first_t0 = time.perf_counter()
        res = process(p, write_overlay, timer=timer)
        if profile and i == 0:
            first_dt = time.perf_counter() - first_t0
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
    if profile:
        print(f"first image {first_dt:.1f}s (includes one-time EasyOCR model load)")
        timer.report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
