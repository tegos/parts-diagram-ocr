<div align="center">
<h1>parts-diagram-ocr</h1>
<p>Detect callout numbers and their grouping braces on exploded-view spare-parts diagrams.</p>

<a href="https://tegos.github.io/parts-diagram-ocr/sample.html">
  <img src="assets/hero.webp" alt="interactive viewer: fit, zoom, pan and click-to-focus a callout" width="800">
</a>

<sub><b><a href="https://tegos.github.io/parts-diagram-ocr/sample.html">▶ Open the live interactive viewer</a></b> — fit · wheel-zoom · drag-pan · click a number to fly to its part</sub>
</div>

---

Exploded-view assembly diagrams from major spare-parts catalogs label each part
with a small **callout number** at the end of a leader line, and group related
parts with a `{` **brace**. This tool reads those callouts and reconstructs the
groupings.

Built for **raster** diagrams: black line art on white (PNG/GIF), the static
format such catalogs export. Vector (SVG) schematics are already clickable and
don't need OCR; this fills the gap for the raster ones.

| Input | Detected |
|:---:|:---:|
| ![input](assets/example-input.png) | ![output](assets/example-overlay.png) |

*Red boxes = detected callout numbers. On diagrams that use grouping braces,
each brace is drawn in magenta with its group id.*

## How it works

Clean black-on-white line art defeats off-the-shelf OCR detectors (they look for
words, not isolated digits), so detection is done with classic CV and only the
*recognition* uses ML:

1. **Detect** glyph blobs with connected components, group adjacent ones into numbers.
2. **Gate** by whitespace isolation: real callouts sit in the margin, drawing
   features sit in ink.
3. **Recognize** every candidate in one batched EasyOCR call (`0-9` + `A/B` suffix).
4. **Group** detected `{` braces (vertical and horizontal), binding each to its
   group-id + member callouts.

See [`docs/adr/`](docs/adr) for the decisions behind this (engine choice, grouping).

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # opencv, numpy, Pillow, easyocr (pulls torch, CPU)
```

## Usage

```bash
# all images in data/images/
python -m src.detect

# specific files
python -m src.detect data/images/258103600.png

# JSON only, no overlay images
python -m src.detect --no-overlay
```

Outputs land in `data/out/`:
- `<name>.json`: detections + groups
- `<name>_overlay.png`: annotated image

```json
{
  "image": "258103600.png",
  "detections": [{"text": "17", "conf": 1.0, "bbox": [135, 340, 179, 374]}],
  "groups": [{"group": "15", "members": [{"text": "17", "bbox": [...]}],
              "brace_bbox": [...], "group_bbox": [...]}]
}
```

Throughput ≈ 1 s/image after a one-time ~30 s model load.

## Interactive viewer

Turn detections into a self-contained HTML page — pan/zoom the diagram, hover a
box to light its callout, click a number in the sidebar to fly to its part
(the animation above). Point it at an image or a result `.json`:

```bash
python -m src.viewer data/images/258103600.png --out docs
# -> docs/258103600.html  (image copied alongside, no server needed)
```

Open the file directly, or publish `docs/` via GitHub Pages — that's how the
[live demo](https://tegos.github.io/parts-diagram-ocr/sample.html) is served.

## Tests

```bash
pip install pytest && python -m pytest tests -q
```

## Limitations

- Interior drawing features occasionally misread as a digit.
- Nested braces double-count: a sub-group's members also list under the outer group.
- Only `A`/`B` callout suffixes; the letter is sometimes lost when faint (`4A` → `4`).
- Very small/thin callouts can be missed.

The original 2018 Python 2.7 template-matching + KNN prototype is kept under
[`legacy/`](legacy) for reference.

## License

MIT. See [LICENSE](LICENSE).
