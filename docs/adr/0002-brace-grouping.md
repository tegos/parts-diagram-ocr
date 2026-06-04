# ADR 0002 — callout grouping via brace detection

**Status:** accepted (Phase 3, 2026-06-04)

## Context
User wants part ↔ number association. ETKA diagrams encode grouping with a tall
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

## Known limitations / future work
- Per-layout variation in ETKA brackets not fully handled.
- `2A`-style alphanumeric ids still digit-only (recognition limitation).
- Optional later: local leader tracing for true digit→part geometry.
