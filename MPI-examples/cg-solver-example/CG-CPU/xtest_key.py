#!/usr/bin/env python3
"""Send keysyms via X11 XTEST, e.g. to expand an SWT tree ('asterisk' = expand
subtree, 'Right'/'Down' to walk it).

Usage:  python3 xtest_key.py <keysym> [repeat]
Env:    DISPLAY must point at the (Xvfb) server.
"""
import sys
import time
from Xlib import X, XK, display
from Xlib.ext import xtest


def main():
    sym = sys.argv[1]
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    d = display.Display()
    code = d.keysym_to_keycode(XK.string_to_keysym(sym))
    for _ in range(repeat):
        xtest.fake_input(d, X.KeyPress, code); d.sync(); time.sleep(0.1)
        xtest.fake_input(d, X.KeyRelease, code); d.sync(); time.sleep(0.25)
    d.sync(); time.sleep(0.3)
    print(f"pressed {sym} x{repeat}")


if __name__ == "__main__":
    main()
