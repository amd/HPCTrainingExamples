#!/usr/bin/env python3
"""Always-on-top caption bar that displays caption.txt.

A borderless, semi-transparent bar pinned to the top or bottom of the screen,
intended to be captured when sharing the entire screen. On Windows the bar is
click-through, so clicks pass to the windows beneath it; press Ctrl+Alt+Q to
quit. On other platforms press Escape.

Environment variables: KAL_OVL_FONT, KAL_OVL_ALPHA, KAL_OVL_H (minimum height),
KAL_OVL_POS (top|bottom), KAL_OVL_MARGIN (gap from the edge),
KAL_OVL_IDLE (seconds of no update before the bar clears; 0 disables),
KAL_OVL_CLICKTHROUGH (set 0 to keep the bar clickable).
"""
import os, time, tkinter as tk

HERE       = os.path.dirname(os.path.abspath(__file__))
CAP_FILE   = os.environ.get("KAL_CAP_FILE", os.path.join(HERE, "caption.txt"))
FONT_SZ    = int(os.environ.get("KAL_OVL_FONT", "18"))
ALPHA      = float(os.environ.get("KAL_OVL_ALPHA", "0.85"))
MIN_H      = int(os.environ.get("KAL_OVL_H", "48"))
MARGIN     = int(os.environ.get("KAL_OVL_MARGIN", "80"))
POS        = os.environ.get("KAL_OVL_POS", "bottom").lower()
IDLE_CLEAR = float(os.environ.get("KAL_OVL_IDLE", "6"))
CLICK_THRU = os.environ.get("KAL_OVL_CLICKTHROUGH", "1") != "0"
IS_WIN     = os.name == "nt"
BG, FG, POLL_MS = "#0b0b0b", "#ffffff", 200

def _set_click_through(root):
    if not (IS_WIN and CLICK_THRU):
        return
    import ctypes
    try:
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        GWL_EXSTYLE, WS_EX_LAYERED, WS_EX_TRANSPARENT = -20, 0x00080000, 0x00000020
        cur = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE,
                                            cur | WS_EX_LAYERED | WS_EX_TRANSPARENT)
    except Exception as e:
        print(f"[overlay] click-through setup failed: {e}")

def _poll_quit(root):
    if not IS_WIN:
        return
    import ctypes
    down = lambda vk: bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)
    if down(0x11) and down(0x12) and down(0x51):        # Ctrl + Alt + Q
        root.destroy()
        return
    root.after(120, lambda: _poll_quit(root))

def main():
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", ALPHA)
    root.configure(bg=BG)
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    lbl = tk.Label(root, text="", bg=BG, fg=FG, font=("Segoe UI", FONT_SZ, "bold"),
                   wraplength=sw - 80, justify="center", anchor="center")
    lbl.pack(expand=True, fill="both", padx=24, pady=8)

    def place(h):                                       # grows toward center as text wraps
        root.geometry(f"{sw}x{h}+0+{MARGIN if POS == 'top' else sh - MARGIN - h}")

    place(MIN_H)
    last = {"txt": None}
    def refresh():
        try:
            with open(CAP_FILE, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if IDLE_CLEAR > 0 and time.time() - os.path.getmtime(CAP_FILE) > IDLE_CLEAR:
                txt = ""                                # idle: clear so stray lines do not linger
        except Exception:
            txt = ""
        if txt != last["txt"]:
            lbl.config(text=txt)
            last["txt"] = txt
            root.update_idletasks()
            place(max(MIN_H, lbl.winfo_reqheight() + 16))
        root.after(POLL_MS, refresh)

    root.bind("<Escape>", lambda e: root.destroy())
    root.after(200, lambda: _set_click_through(root))
    _poll_quit(root)
    refresh()
    root.mainloop()

if __name__ == "__main__":
    main()
