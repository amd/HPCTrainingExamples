#!/usr/bin/env python3
"""Send a sequence of left-clicks via X11 XTEST at the given pixel coordinates.

Usage:  python3 xtest_click.py x1 y1 [x2 y2 ...]
Env:    DISPLAY must point at the (Xvfb) server.
"""
import sys
import time
from Xlib import X, display
from Xlib.ext import xtest


def click(d, x, y):
    xtest.fake_input(d, X.MotionNotify, x=x, y=y); d.sync(); time.sleep(0.3)
    xtest.fake_input(d, X.ButtonPress, 1); d.sync(); time.sleep(0.15)
    xtest.fake_input(d, X.ButtonRelease, 1); d.sync(); time.sleep(0.6)


def main():
    coords = list(map(int, sys.argv[1:]))
    d = display.Display()
    for i in range(0, len(coords) - 1, 2):
        click(d, coords[i], coords[i + 1])
        print(f"clicked ({coords[i]},{coords[i+1]})")
    d.sync(); time.sleep(0.5)


if __name__ == "__main__":
    main()
