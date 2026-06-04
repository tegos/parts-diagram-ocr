# ADR 0001 — OCR approach for callout digits

**Status:** accepted (Phase 1, 2026-06-04)

## Context
Input: exploded-view spare-parts diagrams from major parts catalogs — raster
(PNG) clean black line art on white.
Task: detect callout numbers (1..~99) at the ends of leader lines, grouped by
`{` braces. Numbers are small, isolated, scattered around the periphery.

## Options considered
- **Pure ML OCR (EasyOCR / PaddleOCR / Tesseract).** Spiked EasyOCR on 5 images.
- **Hybrid:** connected-components detection + ML recognition of crops.

## Spike findings (EasyOCR, CPU)
- Dense label clusters: excellent, conf ≈ 1.0 (258103600 → 30 callouts).
- Sparse isolated single digits: **fails** — the CRAFT detector is trained on
  words, not lone glyphs (460127200 → 1 of ~14; only 8 even with aggressive
  thresholds + upscale).
- Hybrid CC + EasyOCR-recognize on the same sparse image → **14** numbers.

## Decision
**Hybrid: connected-components for detection, EasyOCR for recognition only.**
CC trivially finds every black blob on clean line art (high recall); EasyOCR
reads the isolated crop reliably.

## Known tradeoff → Phase 2 precision work
CC over-detects on dense drawing regions (bolts, hatching read as digits:
258103600 → 76 candidates vs ~30 real). Precision levers, all empirically clear:
1. confidence gate (≥ ~0.6)
2. whitespace-isolation gate — real callouts sit in white margin; noise sits
   inside inked drawing regions
3. leader-line / brace association (Phase 3) — real callouts attach to a line;
   internal noise does not

## Update — recovering narrow "1" / "11" (previously a documented limitation)
The CC glyph filter dropped narrow "1" callouts: `GLYPH_MIN_W=6` (a printed "1" is
w4-5 incl. its serif/flag) and `GLYPH_MIN_AREA=50` (a small "1" is ~38px). So "1",
"10", "11" went undetected on some diagrams (the alternator sample missed all three).

An earlier attempt was abandoned as unsafe — it had *also* dropped the fill cap to
admit solid bars, which flooded false "1"s (460127200: 9 → 46) from leader lines,
hidden by the set-based eval. Re-measuring the real glyphs corrected that diagnosis:
a real "1" is **not** a 1px bar. It is w4-5, **fill ≈ 0.5** (serif + stem, not a full
column), and aspect (h/w) ≈ 3-4 at every scale. The flood came from leader-line
fragments, which are either solid (fill ≈ 1.0) or bare bars at aspect 10-13.

Fix: lower `GLYPH_MIN_W` 6→3 and `GLYPH_MIN_AREA` 50→30, and add `GLYPH_MAX_ASPECT=7`
(h/w). Three gates now separate real "1"s from line fragments:
- `GLYPH_MAX_FILL=0.85` — rejects solid 1px bars (leaders, fill ≈ 1.0)
- `ISO_MAX_INK=0.10` — rejects non-isolated strokes inside the drawing
- `GLYPH_MAX_ASPECT=7` — rejects the remaining tall-thin isolated bars (aspect 10+)

Evidence: detections on the 5 eval images are **byte-identical** (precision 100,
recall 97, F1 98 — unchanged; their "1"s were already wide enough). The sample
recovers 1/10/11 (all 11 part numbers). Across all 85 images the relaxed filter
admits only 25 narrow candidates total (max 3/image) — no flood. "11" needs no
special handling: two recovered "1" glyphs merge via the existing `group_numbers`.

## Artifacts
`src/spike_easyocr.py`, `src/spike_hybrid.py` (kept as reference baselines).
