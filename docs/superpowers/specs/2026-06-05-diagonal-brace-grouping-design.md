# Diagonal brace grouping (auto-numbering)

Date: 2026-06-05
Status: approved

## Problem

Diagonal open-line braces are detected (`detect_open_braces`) but never bound
to a group id + members, so diagrams whose grouping is drawn diagonally report
0 groups (`data/sample.png` has 3 real groups: ids 2, 1, 5). Since the cyan
layer was dropped from the static overlay (PR #18), these groups are invisible
there. ADR 0002 left auto-numbering as future work; two of its blockers are now
gone (narrow "1" recovers as a callout; sharp-corner/prong machinery exists
from PR #17).

## Measured geometry (sample.png, all 3 braces)

- The brace is a long diagonal **spine** spanning the grouped parts, with short
  **prongs** (sharp direction changes) off the spine.
- The **group id** callout sits at a prong tip, offset from the spine
  (~38-50 px measured).
- **Member** callouts sit close to the spine (0-20 px measured), within the
  spine's span, on the opposite side from the id prong or on the line itself.

## Design

New `associate_open(open_braces, dets, binv, image_width, image_height)` in
`src/braces.py`, mirroring `associate()`'s output shape:

1. **Spine**: PCA principal axis of the brace component's pixels → unit vector
   `u`, perpendicular `v`, centroid `c`. Span = [min, max] of pixel projections
   onto `u`.
2. **Prong tip**: among approxPolyDP sharp vertices (same parameters as
   `_sharp_corners`: eps 4, edges ≥ 8 px, angle < 140°), the one with max
   |projection on `v`| — the deepest excursion from the spine. Its sign picks
   the id side.
3. **Id**: callout centres within span, on the prong side, within
   `GROUP_REACH × perp_image_dim` of the prong tip → nearest one. None → brace
   stays unbound (JSON `open_braces` only, as today).
4. **Members**: callouts within span whose perpendicular distance to the spine
   line is ≤ `OPEN_MEMBER_REACH × perp_image_dim` (new constant, ~0.04 — members
   hug the spine), excluding the id. None → unbound.
5. Bound braces are appended to the `groups` list in `detect.py` and drawn
   magenta like axis-aligned groups. JSON: same `groups` entry format
   (`group`, `group_bbox`, `brace_bbox`, `members`).

Untouched: `associate()` (axis-aligned path), callout pipeline, eval scoring.

## Expected results

- sample.png: ≥ 2 of 3 groups bound (grp 2 → members 7, 8; grp 1 → members
  3, 4 and possibly nested 5, 6 — nested double-count is a known, documented
  limitation). Grp 5's members have no own callout digits, so it may honestly
  stay unbound.
- No regression: 35 unit tests green, eval precision/recall 100%, axis-aligned
  groups byte-identical.
- Across the 100-image set, new groups appear only where a surviving open brace
  passes the id+members gate; survivors are ~50, mostly part contours, and the
  gate is expected to suppress nearly all of them (verified by eyeballing every
  newly bound group).

## Testing

- Synthetic: pronged polyline + callouts placed per the geometry → one group
  with correct id and members; same polyline with no id-side callout → no
  group; no member callouts → no group.
- Regression: sample.png yields ≥ 2 groups with ids {1, 2} and correct members.
- Existing suite green; eval unchanged.

## Risks

- False groups from surviving part-contour braces: the id+members gate plus
  the narrow member reach (0.04) is the control; verified visually across the
  dataset before merge.
- PCA spine unstable for very curved survivors — acceptable, they then fail
  the gate and stay unbound.
