# legacy/ — 2018 Python 2.7 code (reference only, not run)

Kept for reference while modernizing. The live pipeline is in `src/`.

| Bucket | What | Use now |
|---|---|---|
| `old-pipeline/` | original entry (`cv/main.py`, `functions.py`, `config.py`, `script.py`) — template-matching + KNN flow | overview of the original approach |
| `bracket-detection/` | brace/leader detection (`worked/tm_find_start_bracket.py` etc.) + brace templates (`res/br_start.png`, `br_end.png`) | **reference for Phase 3** (digit ↔ leader/brace association) |
| `knn-recognition/` | old digit recognizer: KNN model (`digits/digits_cls.pkl`) + digit templates (`templates/`) | superseded by EasyOCR |
| `experiments/` | one-off spikes (SIFT, regionprops, contour plots) | low value |

Deleted during cleanup: scratch debug images, a duplicate `digits_cls.pkl`, a
stray output `res.png`. All recoverable from git history.
