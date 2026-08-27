#!/usr/bin/env python3
"""Drive a Qt combo box via X11 XTEST: click at (x,y) to open a popup, then send
`down` ArrowDown keypresses and Return to pick the N-th entry. Robust to the exact
pixel positions of the popup items (keyboard navigation, not item clicks).

Usage:  python3 xtest_select.py <x> <y> <down> [clicks_before]
Env:    DISPLAY must point at the (Xvfb) server.
"""
import sys
import time
from Xlib import X, XK, display
from Xlib.ext import xtest


def main():
    x, y, down = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    clicks = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    d = display.Display()

    def move_click():
        xtest.fake_input(d, X.MotionNotify, x=x, y=y); d.sync()
        time.sleep(0.3)
        xtest.fake_input(d, X.ButtonPress, 1); d.sync(); time.sleep(0.15)
        xtest.fake_input(d, X.ButtonRelease, 1); d.sync(); time.sleep(0.5)

    def key(keysym):
        code = d.keysym_to_keycode(keysym)
        xtest.fake_input(d, X.KeyPress, code); d.sync(); time.sleep(0.12)
        xtest.fake_input(d, X.KeyRelease, code); d.sync(); time.sleep(0.12)

    for _ in range(clicks):
        move_click()
    time.sleep(0.4)
    for _ in range(down):
        key(XK.string_to_keysym("Down"))
    key(XK.string_to_keysym("Return"))
    d.sync()
    time.sleep(0.5)
    print(f"selected via {down} Down + Return at ({x},{y})")


if __name__ == "__main__":
    main()
