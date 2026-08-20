#!/usr/bin/env python3
"""Rubber-band drag via X11 XTEST: press at (x1,y1), move to (x2,y2), release.
Used to zoom a time range in hpcviewer's trace ("Main") view.

Usage:  python3 xtest_drag.py x1 y1 x2 y2 [steps]
Env:    DISPLAY must point at the (Xvfb) server.
"""
import sys
import time
from Xlib import X, display
from Xlib.ext import xtest


def main():
    x1, y1, x2, y2 = (int(v) for v in sys.argv[1:5])
    steps = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    d = display.Display()

    xtest.fake_input(d, X.MotionNotify, x=x1, y=y1); d.sync(); time.sleep(0.3)
    xtest.fake_input(d, X.ButtonPress, 1); d.sync(); time.sleep(0.2)
    for i in range(1, steps + 1):
        x = int(x1 + (x2 - x1) * i / steps)
        y = int(y1 + (y2 - y1) * i / steps)
        xtest.fake_input(d, X.MotionNotify, x=x, y=y); d.sync(); time.sleep(0.03)
    time.sleep(0.2)
    xtest.fake_input(d, X.ButtonRelease, 1); d.sync(); time.sleep(0.8)
    print(f"dragged ({x1},{y1}) -> ({x2},{y2})")


if __name__ == "__main__":
    main()
