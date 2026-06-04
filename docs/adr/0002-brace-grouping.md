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

## Known limitations / future work
- Nested braces double-count: a sub-group's members also appear in the enclosing
  group (194500200: 12A in both grp 6 and grp 21).
- Interior duplicate misreads (one number read 2-3× from drawing features).
- Per-layout variation in catalog brackets not fully handled.
- Alphanumeric ids: A/B suffixes now recognized (allowlist `0-9AB` + `valid_callout`
  regex `\d{1,3}[AB]?`); fixed 1A/1B/16A on 121105250. A occasionally still lost
  when its glyph is faint/detached (4A → 4). Other suffix letters not handled.
- Optional later: local leader tracing for true digit→part geometry.
