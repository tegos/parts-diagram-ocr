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


def member_pr(gt_members, det_members):
    """Member-set precision/recall for one matched group id."""
    gt_s, det_s = set(gt_members), set(det_members)
    if not gt_s and not det_s:
        return 1.0, 1.0
    p = len(gt_s & det_s) / len(det_s) if det_s else 1.0
    r = len(gt_s & det_s) / len(gt_s) if gt_s else 1.0
    return p, r


def score_groups(gt_groups, det_groups):
    """Id-set scoring: detected group is TP when its id is a GT group id.

    gt_groups: {"<id>": ["<member>", ...]} from the GT file.
    det_groups: the detection JSON "groups" list.
    Nested braces can emit one id twice; member sets union before comparison.
    Returns (tp_ids, fp_ids, fn_ids, member_rows[(id, mp, mr)]).
    """
    det_by_id = {}
    for g in det_groups:
        det_by_id.setdefault(g["group"], set()).update(
            m["text"] for m in g["members"])
    gt_ids, det_ids = set(gt_groups), set(det_by_id)
    tp = gt_ids & det_ids
    rows = [(gid, *member_pr(gt_groups[gid], det_by_id[gid]))
            for gid in sorted(tp)]
    return tp, det_ids - gt_ids, gt_ids - tp, rows


def score():
    rows = []
    grows = []
    TP = FP = FN = 0
    GTP = GFP = GFN = 0
    for gt_file in sorted(GT_DIR.glob("*.json")):
        stem = gt_file.stem
        gt_dict = json.loads(gt_file.read_text())
        gt = set(gt_dict["callouts"])
        det_file = OUT_DIR / f"{stem}.json"
        if not det_file.exists():
            print(f"!! no detection output for {stem}; run: python -m src.detect")
            continue
        det_dict = json.loads(det_file.read_text())
        dets = det_dict["detections"]
        det_texts = [d["text"] for d in dets]
        det = set(det_texts)
        tp, fp, fn = gt & det, det - gt, gt - det
        dups = len(det_texts) - len(det)
        TP, FP, FN = TP + len(tp), FP + len(fp), FN + len(fn)
        rows.append((stem, len(gt), len(tp), sorted(fp), sorted(fn), dups))

        # Group scoring if GT has "groups" key
        if "groups" in gt_dict:
            gt_groups = gt_dict.get("groups", {})
            det_groups = det_dict.get("groups", [])
            gtp, gfp, gfn, rows_ = score_groups(gt_groups, det_groups)
            GTP += len(gtp)
            GFP += len(gfp)
            GFN += len(gfn)
            grows.append((stem, sorted(gtp), sorted(gfp), sorted(gfn), rows_))

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

    if grows:
        print(f"\n{'image':<14}group hits / false+ / missed   (member p/r per hit)")
        for stem, tp, fp, fn, rows_ in grows:
            mem = " ".join(f"{gid}:{p:.0%}/{r:.0%}" for gid, p, r in rows_)
            print(f"{stem:<14}{tp}  +{fp}  -{fn}   ({mem})")
        gp = GTP / (GTP + GFP) if GTP + GFP else 0
        gr = GTP / (GTP + GFN) if GTP + GFN else 0
        gf1 = 2 * gp * gr / (gp + gr) if gp + gr else 0
        print(f"GROUPS over {len(grows)} images: "
              f"precision {gp:.0%}  recall {gr:.0%}  F1 {gf1:.0%}")


if __name__ == "__main__":
    score()
