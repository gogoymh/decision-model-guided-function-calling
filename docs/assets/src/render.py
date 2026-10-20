"""Render an HTML file to PNG at 2x.

Usage: python docs/assets/src/render.py docs/assets/src/hero.html docs/assets/dmfc-hero.png 1280 460
"""
import sys
import os
from pathlib import Path
from playwright.sync_api import sync_playwright
src, out, width, height = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE"))
    page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=2)
    page.goto(Path(src).resolve().as_uri())
    page.wait_for_timeout(300)
    page.screenshot(path=out, omit_background=True, full_page=False)
    browser.close()
print(out)
