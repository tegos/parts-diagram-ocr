"""Interactive HTML viewer for a detection result.

Reads a `<name>.json` (detections + open_braces) plus the source image and emits
`<name>.html`: the image with a hover-linked sidebar of callout numbers. Hover a
number to highlight every box where it was found (and hover a box to highlight the
number). The image is referenced as a sibling file (copied next to the html), so
keep the two together.

Usage:
    python -m src.viewer data/sample.png             # -> data/out/sample.html (+ .png)
    python -m src.viewer data/out/258103600.json     # json + sibling image
    python -m src.viewer data/sample.png --out docs  # build into docs/ for GH Pages
"""
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2

from src.config import IMAGES_DIR, OUT_DIR

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ — detections</title>
<style>
  :root { --accent:#e23; --brace:#0a8; --ink:#1a1a1a; --muted:#888; --line:#e7e7e7; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.4 -apple-system,Segoe UI,Roboto,sans-serif;
         color:var(--ink); background:#f4f4f5; }
  header { padding:14px 20px; border-bottom:1px solid var(--line); background:#fff; }
  header h1 { margin:0; font-size:16px; font-weight:600; }
  header span { color:var(--muted); font-weight:400; }
  .wrap { display:flex; gap:0; height:calc(100vh - 51px); }
  aside { width:230px; flex:none; overflow-y:auto; border-right:1px solid var(--line);
          background:#fff; padding:10px; }
  aside h2 { font-size:11px; text-transform:uppercase; letter-spacing:.06em;
             color:var(--muted); margin:14px 6px 6px; }
  .item { display:flex; align-items:center; justify-content:space-between;
          padding:6px 10px; margin:2px 0; border-radius:7px; cursor:pointer;
          border:1px solid transparent; }
  .item:hover, .item.on { background:#fff0f1; border-color:var(--accent); }
  .item .num { font:600 14px ui-monospace,SFMono-Regular,Menlo,monospace; }
  .item .cnt { font-size:11px; color:var(--muted); background:#f0f0f1;
               border-radius:10px; padding:1px 7px; }
  .item.brace:hover, .item.brace.on { background:#e7faf5; border-color:var(--brace); }
  main { flex:1; position:relative; overflow:hidden; padding:0;
         background:#f4f4f5; touch-action:none; }
  .stage { position:absolute; top:0; left:0; transform-origin:0 0; cursor:grab;
           box-shadow:0 1px 8px rgba(0,0,0,.12); background:#fff; will-change:transform; }
  .stage.dragging { cursor:grabbing; }
  .stage.animate { transition:transform .35s cubic-bezier(.22,.61,.36,1); }
  .stage img { display:block; }
  .tools { position:absolute; right:14px; bottom:14px; display:flex; gap:4px;
           background:#fff; border:1px solid var(--line); border-radius:9px;
           padding:4px; box-shadow:0 1px 8px rgba(0,0,0,.14); user-select:none; }
  .tools button { width:30px; height:30px; border:0; background:#fff; cursor:pointer;
           border-radius:6px; font:600 15px ui-monospace,monospace; color:var(--ink); }
  .tools button:hover { background:#f0f0f1; }
  .tools button.fit { width:auto; padding:0 10px; font-size:12px; }
  .box { position:absolute; border:2px solid transparent; border-radius:2px;
         pointer-events:auto; transition:background .08s,border-color .08s; }
  .box.callout { border-color:transparent; }
  .box.callout.on { border-color:var(--accent); background:rgba(238,34,51,.18); }
  .box.brace.on   { border-color:var(--brace);  background:rgba(0,168,136,.14); }
  body.showall .box.callout { border-color:rgba(238,34,51,.35); }
  body.showall .box.brace   { border-color:rgba(0,168,136,.4); }
  .tag { position:absolute; top:-9px; left:-2px; transform:translateY(-100%);
         background:var(--accent); color:#fff; font:600 11px ui-monospace,monospace;
         padding:1px 6px; border-radius:6px; white-space:nowrap; display:none; }
  .box.on .tag { display:block; }
  .box.brace .tag { background:var(--brace); }
  .opt { font-size:12px; color:var(--muted); margin:2px 6px 10px; }
  .opt input { vertical-align:-1px; }
</style>
</head>
<body>
<header><h1>__TITLE__ <span>· __NDET__ callouts · __NBRACE__ braces · __NGROUP__ groups</span></h1></header>
<div class="wrap">
  <aside>
    <label class="opt"><input type="checkbox" id="showall"> show all boxes</label>
    <h2>Callout numbers</h2>
    <div id="list"></div>
    <div id="bracelist"></div>
  </aside>
  <main id="main">
    <div class="stage" id="stage"><img src="__IMG__" alt="diagram"></div>
    <div class="tools">
      <button id="zout" title="Zoom out">−</button>
      <button id="fit" class="fit" title="Fit to view">Fit</button>
      <button id="zin" title="Zoom in">+</button>
    </div>
  </main>
</div>
<script>
const DATA = __DATA__;
const NW = DATA.imgW, NH = DATA.imgH;       // natural image size in px
const main = document.getElementById('main');
const stage = document.getElementById('stage');
const boxEl = {}, boxOf = {}, itemOf = {};  // id -> element, id -> box data, id -> sidebar key
stage.style.width = NW + 'px'; stage.style.height = NH + 'px';

function hi(ids, on) {
  for (const id of ids) {
    if (boxEl[id]) boxEl[id].classList.toggle('on', on);
    const it = document.querySelector('.item[data-key="' + itemOf[id] + '"]');
    if (it) it.classList.toggle('on', on);
  }
}
for (const b of DATA.boxes) {
  const d = document.createElement('div');
  d.className = 'box ' + b.kind;
  d.style.left = b.x + '%'; d.style.top = b.y + '%';
  d.style.width = b.w + '%'; d.style.height = b.h + '%';
  d.innerHTML = '<span class="tag">' + b.text + '</span>';
  stage.appendChild(d);
  boxEl[b.id] = d; boxOf[b.id] = b;
  // siblings = every box sharing this number (so both "7"s light together)
  d.addEventListener('mouseenter', () => hi(b.siblings, true));
  d.addEventListener('mouseleave', () => hi(b.siblings, false));
}

// --- pan / zoom: a single transform on #stage (boxes ride along) ---
let scale = 1, tx = 0, ty = 0, fitScale = 1;
const MAX = 8;
function apply() { stage.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')'; }
function clampScale(s) { return Math.max(fitScale, Math.min(MAX, s)); }
function fit() {
  const r = main.getBoundingClientRect();
  fitScale = Math.min(r.width / NW, r.height / NH) * 0.96;
  scale = fitScale;
  tx = (r.width - NW * scale) / 2;
  ty = (r.height - NH * scale) / 2;
  apply();
}
// zoom around a focal point given in main-client coords
function zoomTo(s, cx, cy) {
  s = clampScale(s);
  const ix = (cx - tx) / scale, iy = (cy - ty) / scale;  // image point under focus
  scale = s; tx = cx - ix * scale; ty = cy - iy * scale;
  apply();
}
function zoomCenter(factor) {
  const r = main.getBoundingClientRect();
  zoomTo(scale * factor, r.width / 2, r.height / 2);
}
// center a box in the viewport (scroll-to-element); zoom in a bit if very small
function focusBox(id) {
  const b = boxOf[id]; if (!b) return;
  const r = main.getBoundingClientRect();
  const px = (b.x + b.w / 2) / 100 * NW, py = (b.y + b.h / 2) / 100 * NH;
  scale = clampScale(Math.max(scale, fitScale * 2.2));
  tx = r.width / 2 - px * scale; ty = r.height / 2 - py * scale;
  stage.classList.add('animate'); apply();
  setTimeout(() => stage.classList.remove('animate'), 400);
}

main.addEventListener('wheel', e => {
  e.preventDefault();
  const r = main.getBoundingClientRect();
  const factor = Math.exp(-e.deltaY * 0.0015);
  zoomTo(scale * factor, e.clientX - r.left, e.clientY - r.top);
}, { passive: false });

let drag = null;
stage.addEventListener('pointerdown', e => {
  drag = { x: e.clientX, y: e.clientY, tx, ty };
  stage.classList.add('dragging'); stage.setPointerCapture(e.pointerId);
});
stage.addEventListener('pointermove', e => {
  if (!drag) return;
  tx = drag.tx + (e.clientX - drag.x); ty = drag.ty + (e.clientY - drag.y); apply();
});
stage.addEventListener('pointerup', e => {
  drag = null; stage.classList.remove('dragging');
  try { stage.releasePointerCapture(e.pointerId); } catch (_) {}
});

document.getElementById('zin').addEventListener('click', () => zoomCenter(1.3));
document.getElementById('zout').addEventListener('click', () => zoomCenter(1 / 1.3));
document.getElementById('fit').addEventListener('click', fit);
let userZoomed = false;
main.addEventListener('wheel', () => userZoomed = true, { passive: true });
stage.addEventListener('pointerdown', () => userZoomed = true);
addEventListener('resize', () => { if (!userZoomed) fit(); });

function makeItem(key, label, ids, kind) {
  const el = document.createElement('div');
  el.className = 'item ' + kind;
  el.dataset.key = key;
  ids.forEach(id => itemOf[id] = key);
  el.innerHTML = '<span class="num">' + label + '</span>' +
                 (ids.length > 1 ? '<span class="cnt">×' + ids.length + '</span>' : '');
  el.addEventListener('mouseenter', () => hi(ids, true));
  el.addEventListener('mouseleave', () => hi(ids, false));
  el.addEventListener('click', () => { userZoomed = true; focusBox(ids[0]); });
  return el;
}
const list = document.getElementById('list');
for (const g of DATA.callouts) list.appendChild(makeItem('c_' + g.text, g.text, g.ids, 'callout'));
const bl = document.getElementById('bracelist');
if (DATA.braces.length) {
  const h = document.createElement('h2'); h.textContent = 'Braces'; bl.appendChild(h);
  DATA.braces.forEach((id, i) => bl.appendChild(makeItem('br_' + id, 'brace ' + (i + 1), [id], 'brace')));
}
document.getElementById('showall').addEventListener('change', e =>
  document.body.classList.toggle('showall', e.target.checked));

fit();
</script>
</body>
</html>
"""


def _sort_key(t):
    m = re.match(r"(\d+)([AB]?)", t)
    return (int(m.group(1)), m.group(2)) if m else (10 ** 6, t)


def build_html(image_path, result, title=None, img_src=None):
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image {image_path}")
    h, w = bgr.shape[:2]
    dets = result.get("detections", [])
    braces = result.get("open_braces", [])
    groups = result.get("groups", [])

    boxes, by_text, brace_ids = [], defaultdict(list), []
    det_id_at = {}   # (text, bbox-tuple) -> box id, to resolve group members
    for i, d in enumerate(dets):
        x0, y0, x1, y1 = d["bbox"]
        bid = f"d{i}"
        boxes.append({"id": bid, "kind": "callout", "text": d["text"],
                      "x": x0 / w * 100, "y": y0 / h * 100,
                      "w": (x1 - x0) / w * 100, "h": (y1 - y0) / h * 100})
        by_text[d["text"]].append(bid)
        det_id_at[(d["text"], tuple(d["bbox"]))] = bid
    for i, b in enumerate(braces):
        bx, by, bw, bh = b
        bid = f"b{i}"
        boxes.append({"id": bid, "kind": "brace", "text": "brace",
                      "x": bx / w * 100, "y": by / h * 100,
                      "w": bw / w * 100, "h": bh / h * 100, "siblings": [bid]})
        brace_ids.append(bid)
    # a callout box's "siblings" = every box sharing its number (light up together)
    for box in boxes:
        if box["kind"] == "callout":
            box["siblings"] = by_text[box["text"]]

    # groups: a magenta label box + (for member groups) a dashed brace box;
    # members resolve to existing detection boxes by exact (text, bbox) match
    group_items = []
    for i, g in enumerate(groups):
        gid = f"g{i}"
        gx0, gy0, gx1, gy1 = g["group_bbox"]
        member_ids = [det_id_at[k] for k in
                      ((m["text"], tuple(m["bbox"])) for m in g["members"])
                      if k in det_id_at]
        gbid = None
        if g["members"]:
            gbid = f"gb{i}"
            bx0, by0, bx1, by1 = g["brace_bbox"]
            boxes.append({"id": gbid, "kind": "groupbrace", "text": f"grp {g['group']}",
                          "x": bx0 / w * 100, "y": by0 / h * 100,
                          "w": (bx1 - bx0) / w * 100, "h": (by1 - by0) / h * 100,
                          "siblings": []})
        sibs = [gid, gbid] if gbid else [gid]
        boxes.append({"id": gid, "kind": "group", "text": f"grp {g['group']}",
                      "x": gx0 / w * 100, "y": gy0 / h * 100,
                      "w": (gx1 - gx0) / w * 100, "h": (gy1 - gy0) / h * 100,
                      "siblings": sibs})
        group_items.append({"id": gid, "brace": gbid, "label": g["group"],
                            "members": member_ids})

    data = {
        "imgW": w, "imgH": h,
        "boxes": boxes,
        "callouts": [{"text": t, "ids": ids}
                     for t, ids in sorted(by_text.items(), key=lambda kv: _sort_key(kv[0]))],
        "braces": brace_ids,
        "groups": group_items,
    }
    src = img_src or Path(image_path).name
    title = title or Path(image_path).name
    return (_TEMPLATE
            .replace("__DATA__", json.dumps(data))
            .replace("__IMG__", src)
            .replace("__TITLE__", title)
            .replace("__NDET__", str(len(dets)))
            .replace("__NBRACE__", str(len(braces)))
            .replace("__NGROUP__", str(len(groups))))


def _resolve(arg):
    """Accept an image path or a result-json path; return (image_path, result)."""
    p = Path(arg)
    if p.suffix == ".json":
        result = json.loads(p.read_text())
        stem = p.stem
        for cand in (IMAGES_DIR / f"{stem}.png", Path("data") / f"{stem}.png"):
            if cand.exists():
                return cand, result
        raise FileNotFoundError(f"no image found for {p}")
    jf = OUT_DIR / f"{p.stem}.json"
    result = json.loads(jf.read_text()) if jf.exists() else {"detections": []}
    return p, result


def main(argv):
    argv = list(argv)
    out_dir = OUT_DIR
    if "--out" in argv:
        i = argv.index("--out")
        try:
            out_dir = Path(argv[i + 1])
        except IndexError:
            print("--out needs a directory", file=sys.stderr)
            return 1
        del argv[i:i + 2]
    if not argv:
        print("Give an image path or a result .json", file=sys.stderr)
        return 1
    out_dir.mkdir(parents=True, exist_ok=True)
    for arg in argv:
        image_path, result = _resolve(arg)
        img_name = Path(image_path).name
        html = build_html(image_path, result, img_src=img_name)
        shutil.copyfile(image_path, out_dir / img_name)  # image rides alongside the html
        out = out_dir / f"{Path(image_path).stem}.html"
        out.write_text(html, encoding="utf-8")
        print(f"{img_name}: "
              f"{len(result.get('detections', []))} callouts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
