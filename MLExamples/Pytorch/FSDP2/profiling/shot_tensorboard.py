#!/usr/bin/env python3
"""Headless TensorBoard (torch-tb-profiler) screenshots for the FSDP2 traces.

Starts TensorBoard on --logdir, waits for the port, opens the PyTorch Profiler
plugin, screenshots the Overview, then tries to switch the "Views" dropdown to the
Distributed view (the comm/computation overlap across ranks) and screenshots that.

Usage: python shot_tensorboard.py <logdir> <out_overview.png> <out_second.png> [port]
"""
import os, sys, time, subprocess, urllib.request

logdir = sys.argv[1] if len(sys.argv) > 1 else "./torch_prof"
out1 = sys.argv[2] if len(sys.argv) > 2 else "figs/fsdp2_tb_overview.png"
out2 = sys.argv[3] if len(sys.argv) > 3 else "figs/fsdp2_tb_trace.png"
port = sys.argv[4] if len(sys.argv) > 4 else "6007"

tb = subprocess.Popen(["tensorboard", "--logdir", logdir, "--port", port,
                       "--bind_all", "--load_fast=false"],
                      stdout=open("tb.log", "w"), stderr=subprocess.STDOUT)
try:
    base = f"http://localhost:{port}"
    for _ in range(40):
        try:
            urllib.request.urlopen(base, timeout=2); break
        except Exception:
            time.sleep(2)

    from playwright.sync_api import sync_playwright
    chrome = os.environ.get("CHROME_BIN") or None
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=chrome, headless=True,
                              args=["--no-sandbox", "--disable-dev-shm-usage"])
        pg = b.new_context(viewport={"width": 1600, "height": 1000},
                           device_scale_factor=1.5).new_page()
        pg.goto(f"{base}/#pytorch_profiler", wait_until="load")
        # torch-tb-profiler loads the run list then renders Overview asynchronously
        time.sleep(25)
        pg.screenshot(path=out1, full_page=True)
        print("wrote", out1)

        # The plugin renders inside an iframe; find it and print frame urls.
        fr = None
        for f in pg.frames:
            u = f.url or ""
            if "pytorch_profiler" in u or "plugin" in u:
                fr = f
            print("frame:", (f.name or "-"), u[:90])
        fl = pg.frame_locator("iframe")
        # Try to switch the "Views" dropdown to Distributed (best comm view).
        for label in ("Distributed", "Trace", "Kernel"):
            try:
                # Ant Design Select: current values are .ant-select-selection-item
                # order in the left nav: Runs(0), Views(1), Workers(2)
                fl.locator(".ant-select-selection-item").nth(1).click(timeout=6000)
                time.sleep(1)
                fl.locator(f".ant-select-item-option[title='{label}']").click(timeout=6000)
                time.sleep(15)
                pg.screenshot(path=out2, full_page=True)
                print("wrote", out2, "(", label, ")")
                break
            except Exception as e:
                print("view switch", label, "failed:", str(e)[:90])
        b.close()
finally:
    tb.terminate()
