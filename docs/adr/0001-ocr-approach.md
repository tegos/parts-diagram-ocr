# ADR 0001 — OCR approach for callout digits

**Status:** accepted (Phase 1, 2026-06-04)

## Context
Input: ETKA exploded-view auto-parts diagrams — clean black line art on white.
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

## Artifacts
`src/spike_easyocr.py`, `src/spike_hybrid.py` (kept as reference baselines).
