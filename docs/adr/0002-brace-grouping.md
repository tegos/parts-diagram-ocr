# ADR 0002 — callout grouping via brace detection

**Status:** accepted (Phase 3, 2026-06-04)

## Context
User wants part ↔ number association. Parts-catalog diagrams encode grouping with a tall
`{` brace: a group-id callout on the brace's outer side, member callouts in the
adjacent column on the inner side.

## Options considered
- **Leader-line tracing** (digit → physical part): spiked global Canny + HoughLinesP.
  Failed — the parts *are* line drawings, so 500-800 lines per image; leaders are
  indistinguishable from part contours (`src/spike_leader.py`). Brittle, abandoned.
- **Brace grouping**: detect braces, bind nearby callouts. Chosen.

## Decision
Detect braces as tall, thin, whitespace-isolated connected components
(`src/braces.py: detect_braces`), then associate (`associate`):
- brace opens toward image centre (left-margin → opens right, etc.)
- group-id = outer-side callout near brace mid-height, just outside the spine
- members = inner-side callouts within the vertical span AND the adjacent column
  (horizontal cap `MEMBER_MAX_DX`, fraction of image width)

The whitespace-isolation gate (reused from Phase 2) rejects part-contour
verticals that are otherwise tall and thin (460127200: 4 → 2 candidates).

## Results (spike + integration, 3 images)
- Brace detection: accurate and clean (258103600: 5/5 braces on the real groups).
- Member binding: **approximate v1** — reliably gets the adjacent column but can
  miss a member or pick a wrong group-id when columns are ambiguous or layouts
  differ. Good enough to surface structure; not a perfect catalog reconstruction.

## Update (manual review)
- Both orientations now handled: vertical `{` and horizontal braces. Association is
  axis-agnostic (member side = side with more aligned callouts; id = lone callout
  near the tip opposite). 194500200 finds grp 21 ({6,7,5}) and grp 6 ({17,12A}).
- Groups without a resolvable id are dropped.

## Update (Phase 3b — diagonal open-line braces)
`detect_braces` only catches axis-aligned braces: it gates on bbox aspect
(`long/short ≥ 3`). On perspective exploded views the grouping bracket is drawn
as a **diagonal** thin line whose bbox is near-square (aspect ≈ 1.7), so it was
missed entirely (`data/sample.png` alternator → 0 of 3 braces).

Added `detect_braces`'s sibling `detect_open_braces`: a brace candidate is a
connected component that is thin (`fill ≤ 0.06`), long (`bbox max side ≥ 80`),
whitespace-isolated (ring ≤ 0.06), and **topologically open** (`holes == 0`).
The hole count is the key discriminator — a grouping bracket is an open polyline,
whereas part contours are closed loops (the rear cover reads 14 holes). On the
sample this yields exactly the 3 braces and nothing else.

**Detection only — not auto-numbered.** These braces are drawn on the overlay
(cyan, label `brace`) and recorded under the JSON `open_braces` key, but are NOT
turned into numbered groups. Reasons, each verified on the sample:
- The prong/notch pointing at the group-id is shallow (~15px); PCA max-deviation
  lands on a line endpoint, not the prong, so the id is mislocated.
- Leader lines run flush against the brace, so nearest-callout picks the leader's
  number (big diagonal → "7") instead of the real id.
- The id "1" sits at the prong but is a narrow digit; it is now recovered as a
  callout (ADR 0001 update), yet pinning it to *this* brace still needs the prong
  geometry above, which remains the blocker.
- The open-line filter floods on other layouts (121105250 → 10, 194500200 → 13
  candidates); only the strict `associate` id+members gate keeps those inert.
  Looser id rules would spawn false groups (guarded: groups on the 5 eval images
  stay unchanged, 5→5 / 2→2).

Net: detection is safe and regression-free; reliable numbering of diagonal
labeled-bracket groups needs a genuine prong/leader discriminator (or targeted
recognition at the prong tip to recover the id, including the thin "1"), left as
future work.

## Update (open-line filter tightened — prongs required)
The flood noted above was real: across the 100-image set the open-line filter
reported 402 boxes on 83 images, and on 121105250 all 10 were wrong (5 plain
leader lines, 3 concentric flywheel-rim arc fragments, 2 duplicates of braces
the axis-aligned detector had already bound to groups). Three additions, each
threshold measured on real true/false candidates:

- **Sharp corners ≥ 5** (`approxPolyDP`, edges ≥ 8px, angle < 140°): a bracket
  has prongs — that's its grouping semantics; a straight leader line doesn't.
  Measured: real braces 7–76 corners, leader lines 1–3. (This redefines the
  contract: a plain diagonal stroke is now correctly *rejected*.)
- **Mutual bbox-overlap suppression (IoU ≥ 0.35)**: candidates overlapping each
  other are fragments of one drawing entity (flywheel arcs measured pairwise
  0.37–0.7; distinct real braces with nested bboxes measured 0.015).
- **`exclude=` dedup**: candidates overlapping a `detect_braces` box are the
  same brace caught twice and were double-drawn (cyan over magenta).

Result: 402 → 50 boxes (34 images); sample keeps its 3, 121105250 drops to 0.
Residual: ~2/3 of the surviving 50 are still part-contour fragments (gaskets,
glass panels, wheel arches, wavy valve-cover edges read as pronged polylines) —
real progress now needs brace ground truth, not heuristics.

**Consequence: cyan dropped from the static overlay.** With precision stuck at
~1/3, drawing open braces unconditionally misleads more than it helps. They
remain in the JSON (`open_braces`) and in the interactive viewer, which shows
them on demand (hover / showall) rather than permanently.

## Update (diagonal braces auto-numbered — `associate_open`)
The Phase 3b blockers fell: the narrow "1" reads as a callout (ADR 0001), and
the prong machinery (sharp corners) locates the id notch. Measured on every
true diagonal brace (5 images, 12 braces), the geometry is uniform **in digit
heights** — the only scale stable across catalogs (image-dim fractions broke:
a 0.03×dim member band swallowed the id on large diagrams):

- id sits 0.7–1.0 digit-heights from the prong tip (false bindings: 1.5–4.8);
  gate `OPEN_ID_REACH = 1.3`;
- members hug the spine at 0.6–1.6 on the opposite side from the id
  (unrelated callouts: 8+); gate `OPEN_MEMBER_REACH = 4.5`;
- the prong tip is the deepest sharp vertex in the **central 80 % of the
  span** — stroke end-caps masquerade as deep vertices (the PCA-endpoint trap
  Phase 3b documented);
- the brace's component is matched by **exact bbox** in the full-image
  labeling — "largest component in the box" grabbed part contours overlapping
  big braces' bboxes.

Members may be empty: some braces group parts that carry no callout digits of
their own (sample grp 5 — washers/nuts identified only by the group id), so an
id alone binds. sample.png reports all 3 of its groups.

The 2 false bindings this initially cost (645845000 — a door-glass contour
with a stray "6" near a corner; 258133860 — a hose-clamp fragment with "2"
adjacent) are now gated out by `OPEN_BIND_MIN_SHARP = 5`: a true grouping
bracket is pronged along its whole spine — every true diagonal brace on the
20-image GT set shows 6–9 sharp vertices in the central 80 % of the span
(sample.png: 6/6/10), while both false part-contour bindings showed exactly 3.
Verified across all 100 images: exactly those 2 groups disappear, no true
binding lost.

## Update (group ground truth + eval)
`eval/groundtruth/` now covers 20 images and carries a `groups` key
(`{"<id>": ["<member>", ...]}` as drawn), scored by `eval/score.py` next to
the callout metrics. Baseline after the sharp-vertex gate: group id
precision 83 %, recall 69 %. The eval surfaced defect classes beyond the two
fixed diagonal false groups, all axis-aligned-path issues:

- **Fragment binds**: a dash-dot leader segment (121105250 → "grp 14"), a pipe
  curve (702201090 → "grp 1"), even a bolt shaft (194103550 → "grp 10") pass
  `detect_braces`'s thin-isolated gates and bind nearby labels. A corner gate
  can't cut them: small true V-braces measure 1 sharp corner, same as the
  fragments.
- **Id-side swaps on real braces**: when the id label sits on the member side
  visually (24{25} read as 25{24,27}; 12{14} as 14{12}; 2{3} as 3{2};
  6{8}/5{8} as 8{6}/8{5}; 15A{22} as 22{15A}), `associate` picks the wrong
  side. The brace is right, the id is not.
- **Missed id-only V-braces**: short brackets grouping unlabeled parts
  (656311410 has four: 9, 15, 18, 25) fall under `BRACE_MIN_LONG` or read as
  1-corner strokes.
- **Missed nested diagonal pairs**: 710407300's 22/23 chained braces bind
  neither.

These are recall/member-precision costs, documented and measured; fixing the
id-side rule is the highest-value next step.

## Known limitations / future work
- Axis-aligned association: fragment binds, id-side swaps, missed small
  V-braces (see group-eval update above) — the dominant group-eval cost now.
- Nested braces double-count: a sub-group's members also appear in the enclosing
  group (194500200: 12A in both grp 6 and grp 21).
- Interior duplicate misreads (one number read 2-3× from drawing features).
- Per-layout variation in catalog brackets not fully handled.
- Alphanumeric ids: A/B suffixes now recognized (allowlist `0-9AB` + `valid_callout`
  regex `\d{1,3}[AB]?`); fixed 1A/1B/16A on 121105250. A occasionally still lost
  when its glyph is faint/detached (4A → 4). Other suffix letters not handled.
- Optional later: local leader tracing for true digit→part geometry.
