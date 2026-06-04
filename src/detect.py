"""CLI entry point. Phase 0: validates structure and lists the work set.

OCR detection lands in Phase 2 (engine chosen by the Phase 1 spike).

Usage:
    python -m src.detect                 # all images in data/images
    python -m src.detect data/images/258103600.png ...
"""
import sys
from pathlib import Path

import cv2

from src.config import IMAGES_DIR, OUT_DIR


def collect(args: list[str]) -> list[Path]:
    if args:
        return [Path(a) for a in args]
    return sorted(IMAGES_DIR.glob("*.png"))


def main(argv: list[str]) -> int:
    paths = collect(argv)
    if not paths:
        print(f"No images found in {IMAGES_DIR}", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"SKIP (unreadable): {p}", file=sys.stderr)
            continue
        h, w = img.shape[:2]
        print(f"{p.name}\t{w}x{h}")
        # TODO Phase 2: run OCR -> filter digits -> Phase 3: associate to braces

    print(f"\n{len(paths)} image(s) ready. OCR pipeline pending (Phase 2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
