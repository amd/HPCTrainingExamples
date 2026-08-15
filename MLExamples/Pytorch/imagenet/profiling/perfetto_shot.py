#!/usr/bin/env python3
"""Best-effort headless screenshot of the Perfetto UI with a real trace loaded.

This is intentionally fragile: it needs a bundled Chromium (installed via
`playwright install chromium`) and outbound network to https://ui.perfetto.dev.
It loads the trace into Perfetto using the documented postMessage "open trace"
handshake, waits for the timeline to render, and screenshots it.

Exit codes:
  0  screenshot written to --out
  2  playwright not available
  3  chromium/launch failure
  4  Perfetto did not load the trace in time

On any non-zero exit the caller should fall back to dropping the raw trace file
for manual download + screenshot.
"""
import argparse
import os
import sys
import threading
import functools
import http.server
import socketserver


class _CORSHandler(http.server.SimpleHTTPRequestHandler):
    """Serve the trace dir with permissive CORS so Perfetto can fetch it."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):  # quiet
        pass


def _serve(directory):
    handler = functools.partial(_CORSHandler, directory=directory)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


# JS run in the page: fetch the trace and hand it to Perfetto via postMessage,
# following https://perfetto.dev/docs/visualization/deep-linking-to-perfetto-ui
_OPEN_JS = r"""
async (traceUrl) => {
  const resp = await fetch(traceUrl);
  const buf = await resp.arrayBuffer();
  const win = window;  // the Perfetto UI window (same page)
  // Perfetto answers 'PING' with 'PONG' once its service worker is ready.
  await new Promise((resolve) => {
    const onMsg = (e) => {
      if (e.data === 'PONG') { window.removeEventListener('message', onMsg); resolve(); }
    };
    window.addEventListener('message', onMsg);
    const ping = setInterval(() => win.postMessage('PING', '*'), 250);
    setTimeout(() => { clearInterval(ping); }, 8000);
    // Also resolve after a grace period even if PONG is missed.
    setTimeout(resolve, 8500);
  });
  win.postMessage({ perfetto: { buffer: buf, title: 'imagenet DDP 4xMI300A' } }, '*');
  return buf.byteLength;
}
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("trace", help="Chrome/Perfetto JSON trace to load")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--ui", default="https://ui.perfetto.dev",
                    help="Perfetto UI base URL")
    ap.add_argument("--wait", type=int, default=45,
                    help="seconds to wait for the timeline to render")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.trace):
        print(f"perfetto_shot: no such trace {args.trace}", file=sys.stderr)
        return 4

    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:  # noqa: BLE001
        print(f"perfetto_shot: playwright unavailable ({exc})", file=sys.stderr)
        return 2

    tdir = os.path.dirname(os.path.abspath(args.trace)) or "."
    tname = os.path.basename(args.trace)
    httpd, port = _serve(tdir)
    trace_url = f"http://127.0.0.1:{port}/{tname}"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    try:
        with sync_playwright() as p:
            try:
                browser = p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage",
                          "--use-gl=swiftshader", "--enable-webgl",
                          "--ignore-gpu-blocklist"])
            except Exception as exc:  # noqa: BLE001
                print(f"perfetto_shot: chromium launch failed ({exc})", file=sys.stderr)
                return 3
            page = browser.new_page(viewport={"width": 1600, "height": 900})
            page.set_default_timeout(args.wait * 1000)
            try:
                page.goto(args.ui, wait_until="load", timeout=args.wait * 1000)
            except Exception as exc:  # noqa: BLE001
                print(f"perfetto_shot: could not reach {args.ui} ({exc})",
                      file=sys.stderr)
                browser.close()
                return 4
            try:
                nbytes = page.evaluate(_OPEN_JS, trace_url)
                print(f"perfetto_shot: posted {nbytes} bytes to Perfetto")
            except Exception as exc:  # noqa: BLE001
                print(f"perfetto_shot: postMessage load failed ({exc})",
                      file=sys.stderr)
                browser.close()
                return 4
            # Wait for the timeline canvas to appear, then let it settle.
            ok = False
            try:
                page.wait_for_selector("canvas", timeout=args.wait * 1000)
                ok = True
            except Exception:  # noqa: BLE001
                pass
            page.wait_for_timeout(6000)
            page.screenshot(path=args.out, full_page=False)
            browser.close()
            if not ok:
                print("perfetto_shot: canvas never appeared; screenshot may be blank",
                      file=sys.stderr)
                return 4
            print(f"perfetto_shot: wrote {args.out}")
            return 0
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
