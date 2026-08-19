#!/usr/bin/env python3
"""Load a .pftrace into the real Perfetto UI (ui.perfetto.dev) with headless
Google Chrome and screenshot the timeline.  Uses Perfetto's official
window.postMessage deep-linking API (opener page opens the UI in a popup, waits
for the PING/PONG handshake, then posts the trace ArrayBuffer) so there is no
CORS / mixed-content / local-server dependency.

Usage:
  python screenshot_perfetto.py <trace.pftrace> <out.png> [title] [wait_ms]
Requires: playwright (pip), a reachable ui.perfetto.dev, and google-chrome.
"""
import base64, os, sys, glob

CHROME = os.environ.get("CHROME_BIN") or \
    "/nfsapps/ubuntu-24.04/opt/google-chrome-vstable/bin/google-chrome"

def main():
    trace = sys.argv[1]
    out   = sys.argv[2]
    title = sys.argv[3] if len(sys.argv) > 3 else os.path.basename(trace)
    wait  = int(sys.argv[4]) if len(sys.argv) > 4 else 14000

    b64 = base64.b64encode(open(trace, "rb").read()).decode()
    print(f"trace {trace} ({len(b64)} b64 chars) -> {out}")

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME, headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage",
                  "--use-gl=angle", "--use-angle=swiftshader",
                  "--enable-unsafe-swiftshader", "--hide-scrollbars"])
        ctx = browser.new_context(viewport={"width": 1920, "height": 1000},
                                  device_scale_factor=2)
        opener = ctx.new_page()
        opener.goto("about:blank")
        opener.evaluate("(b64) => { window.__b64 = b64; }", b64)

        with ctx.expect_page() as pop:
            opener.evaluate("() => { window.__win = window.open('https://ui.perfetto.dev'); }")
        ui = pop.value
        ui.wait_for_load_state("domcontentloaded")

        # PING/PONG handshake, then post the trace buffer (runs in opener context,
        # which owns the popup handle and the base64 payload).
        opener.evaluate("""(title) => new Promise((resolve) => {
            const win = window.__win;
            const bin = atob(window.__b64);
            const buf = new Uint8Array(bin.length);
            for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
            const onMsg = (e) => {
                if (e.data !== 'PONG') return;
                clearInterval(t);
                window.removeEventListener('message', onMsg);
                win.postMessage({perfetto: {buffer: buf.buffer, title: title,
                                            fileName: title + '.pftrace'}}, '*');
                resolve(true);
            };
            window.addEventListener('message', onMsg);
            const t = setInterval(() => win.postMessage('PING', 'https://ui.perfetto.dev'), 100);
        })""", title)

        # Perfetto asks to confirm loading a trace from an "unknown origin".
        try:
            ui.locator("button:has-text('Yes')").click(timeout=8000)
            print("clicked 'Yes' on open-trace confirmation")
        except Exception as e:
            print("no confirmation dialog:", e)
        # Dismiss the cookie banner if present (it overlaps the bottom tracks).
        try:
            ui.locator("button:has-text('OK')").click(timeout=3000)
        except Exception:
            pass

        ui.wait_for_timeout(wait)   # let trace_processor parse + render the tracks

        def omnibox(text, mode_prefix=""):
            """Type into the Perfetto omnibox; mode_prefix '>' = command, '' = search."""
            box = ui.locator("input[placeholder*='Search']").first
            box.click(timeout=8000)
            box.fill(mode_prefix + text)
            ui.wait_for_timeout(600)
            ui.keyboard.press("Enter")
            ui.wait_for_timeout(1200)

        # Expand all track groups so the GPU-kernel + roctx marker rows are visible.
        try:
            omnibox("Expand all", ">")
            ui.keyboard.press("Escape")
            ui.wait_for_timeout(500)
        except Exception as e:
            print("expand-all failed:", e)

        # Search for a roctx marker label -> Perfetto selects that slice and pans to
        # it.  Escape leaves the omnibox (selection persists), 'f' zooms to it, then
        # 's' zooms back out a little to reveal the neighbouring iterations.
        # argv: 5=label 6=zoom_out_presses.
        label   = sys.argv[5] if len(sys.argv) > 5 else "cg_iteration"
        zoomout = int(sys.argv[6]) if len(sys.argv) > 6 else 4
        if label and label != "-":
            try:
                omnibox(label, "")
                ui.keyboard.press("Enter")            # step to first match + select
                ui.wait_for_timeout(800)
                ui.evaluate("() => document.activeElement && document.activeElement.blur()")
                ui.wait_for_timeout(400)
                ui.screenshot(path=out + ".pre.png")  # inspect selection/pan state
                ui.keyboard.press("f")                # zoom+pan to the selected slice
                ui.wait_for_timeout(1200)
                for _ in range(zoomout):              # widen to a couple of iterations
                    ui.keyboard.press("s")
                    ui.wait_for_timeout(150)
                ui.wait_for_timeout(800)
            except Exception as e:
                print("search/zoom failed:", e)

        clip_h = int(os.environ.get("PF_CLIP_H", "0"))
        if clip_h > 0:
            ui.screenshot(path=out, clip={"x": 0, "y": 0, "width": 1920, "height": clip_h})
        else:
            ui.screenshot(path=out)
        print("wrote", out)
        browser.close()

if __name__ == "__main__":
    main()
