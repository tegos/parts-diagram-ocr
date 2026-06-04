"""Set-based evaluation: distinct callout numbers, ground-truth vs detected.

Ground truth (eval/groundtruth/<stem>.json) lists the DISTINCT callout texts
visible on a diagram. We score the set of distinct detected texts against it, so
this measures "did we read the right part numbers" — not per-instance counts.
Duplicate misreads (same number found 2-3x) are reported separately.

Usage:
    python -m eval.score
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_DIR = Path(__file__).resolve().parent / "groundtruth"
OUT_DIR = ROOT / "data" / "out"


def score():
    rows = []
    TP = FP = FN = 0
    for gt_file in sorted(GT_DIR.glob("*.json")):
        stem = gt_file.stem
        gt = set(json.loads(gt_file.read_text())["callouts"])
        det_file = OUT_DIR / f"{stem}.json"
        if not det_file.exists():
            print(f"!! no detection output for {stem}; run: python -m src.detect")
            continue
        dets = json.loads(det_file.read_text())["detections"]
        det_texts = [d["text"] for d in dets]
        det = set(det_texts)
        tp, fp, fn = gt & det, det - gt, gt - det
        dups = len(det_texts) - len(det)
        TP, FP, FN = TP + len(tp), FP + len(fp), FN + len(fn)
        rows.append((stem, len(gt), len(tp), sorted(fp), sorted(fn), dups))

    print(f"{'image':<14}{'gt':>4}{'hit':>5}  {'recall':>7}  false+ / missed (dup reads)")
    for stem, n, hit, fp, fn, dups in rows:
        r = hit / n if n else 0
        print(f"{stem:<14}{n:>4}{hit:>5}  {r:>6.0%}   +{fp} -{fn}  (dups {dups})")

    prec = TP / (TP + FP) if TP + FP else 0
    rec = TP / (TP + FN) if TP + FN else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"\nMICRO over {len(rows)} images: "
          f"precision {prec:.0%}  recall {rec:.0%}  F1 {f1:.0%}")
    print("(set-based on distinct numbers; precision ignores duplicate misreads)")


if __name__ == "__main__":
    score()
