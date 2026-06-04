"""Headless self-verify for the HTML viewer.

Loads a built viewer page in headless Chromium, exercises the interactions
(fit -> zoom -> pan -> click-to-center) and asserts the #stage transform reacts
the way it should. Saves a screenshot after each step so a human (or agent) can
eyeball the result.

Usage:
    .venv/bin/python scripts/verify_viewer.py docs/sample.html
    .venv/bin/python scripts/verify_viewer.py docs/sample.html --shots .verify-shots

Setup once: .venv/bin/pip install -r requirements-dev.txt   (chromium already cached)

Exit 0 = all checks pass.
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

W, H = 1280, 800


def _matrix(page):
    """Return (scale, tx, ty) parsed from #stage's computed transform."""
    m = page.evaluate(
        "getComputedStyle(document.getElementById('stage')).transform")
    if not m or m == "none":
        return (1.0, 0.0, 0.0)
    nums = [float(x) for x in m[m.index("(") + 1:m.index(")")].split(",")]
    return (nums[0], nums[4], nums[5])  # a, e, f


def verify(html_path, shots_dir):
    html_path = Path(html_path).resolve()
    shots_dir = Path(shots_dir)
    shots_dir.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, ok):
        checks.append((name, ok))
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": W, "height": H})
        page.goto(html_path.as_uri())
        page.wait_for_selector(".box")
        page.wait_for_timeout(150)  # let fit() settle

        s0, tx0, ty0 = _matrix(page)
        page.screenshot(path=str(shots_dir / "1-fit.png"))
        check("fit applied (scale < 1, image shrunk to viewport)", 0 < s0 < 1)
        check("stage centered (left margin > 0)", tx0 > 0)

        # wheel zoom at a point left-of-center; scale must grow
        page.mouse.move(W * 0.4, H * 0.5)
        page.mouse.wheel(0, -600)
        page.wait_for_timeout(80)
        s1, tx1, ty1 = _matrix(page)
        page.screenshot(path=str(shots_dir / "2-zoom.png"))
        check("wheel zoomed in (scale grew)", s1 > s0 + 1e-3)

        # drag-pan: tx must change
        page.mouse.move(W * 0.5, H * 0.5)
        page.mouse.down()
        page.mouse.move(W * 0.5 + 120, H * 0.5 + 60, steps=5)
        page.mouse.up()
        page.wait_for_timeout(80)
        s2, tx2, ty2 = _matrix(page)
        page.screenshot(path=str(shots_dir / "3-pan.png"))
        check("drag panned (tx moved ~+120)", abs(tx2 - tx1 - 120) < 25)

        # click first sidebar number -> center on its box (transform jumps)
        page.click(".item")
        page.wait_for_timeout(450)  # animation
        s3, tx3, ty3 = _matrix(page)
        page.screenshot(path=str(shots_dir / "4-focus.png"))
        check("click-to-center moved stage",
              abs(tx3 - tx2) > 1 or abs(ty3 - ty2) > 1)

        # Fit button resets to initial fit
        page.click("#fit")
        page.wait_for_timeout(80)
        s4, tx4, ty4 = _matrix(page)
        check("Fit button reset scale", abs(s4 - s0) < 1e-3)

        browser.close()

    ok = all(c[1] for c in checks)
    print(f"\n{'ALL PASS' if ok else 'FAILED'} — shots in {shots_dir}")
    return 0 if ok else 1


def main(argv):
    if not argv:
        print("Give a built viewer .html path", file=sys.stderr)
        return 1
    html = argv[0]
    shots = ".verify-shots"
    if "--shots" in argv:
        shots = argv[argv.index("--shots") + 1]
    return verify(html, shots)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
