#!/usr/bin/env python3
"""Screenshot the roofline-extractor `counters.html` (self-contained d3 plot) with
headless Google Chrome via Playwright.  Optionally click UI buttons first (by
visible text, e.g. "Light mode", "Toggle curved rooflines") and/or a kernel legend
row to isolate one kernel.

Usage:
  python3 screenshot_roofline.py <counters.html> <out.png> [button_text ...]
Env: CHROME_BIN (defaults to the AAC6 google-chrome path).
"""
import os, sys

CHROME = os.environ.get("CHROME_BIN") or \
    "/nfsapps/ubuntu-24.04/opt/google-chrome-vstable/bin/google-chrome"


def main():
    html = os.path.abspath(sys.argv[1])
    out = sys.argv[2]
    buttons = sys.argv[3:]
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME, headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--hide-scrollbars",
                  "--force-device-scale-factor=1.5"])
        ctx = browser.new_context(viewport={"width": 1500, "height": 1050},
                                  device_scale_factor=1.5)
        pg = ctx.new_page()
        pg.goto("file://" + html)
        pg.wait_for_timeout(3500)              # let d3 build the SVG
        for b in buttons:
            try:
                pg.locator(f"button:has-text('{b}')").first.click(timeout=5000)
                pg.wait_for_timeout(1200)
                print("clicked", b)
            except Exception as e:
                print("click failed", b, e)
        pg.wait_for_timeout(800)
        pg.screenshot(path=out)
        print("wrote", out)
        browser.close()


if __name__ == "__main__":
    main()
