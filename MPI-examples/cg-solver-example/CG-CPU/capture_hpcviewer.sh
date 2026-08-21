#!/bin/bash
# Headless screen capture of HPCToolkit's hpcviewer (Eclipse/SWT trace + profile
# view) on a virtual X display (Xvfb), grabbed to PNG with xwd + xwd2png.py.
#   Usage: capture_hpcviewer.sh <database.d> <out.png> [WxH] [wait_s] [display]
# Optional XTEST clicks to switch to the Trace view tab / select a rank:
#   HV_CLICKS="x1 y1 x2 y2 ..."  (see xtest_click.py)
set -u
DB=${1:?database.d}; OUT=${2:?out.png}; GEO=${3:-1920x1150}; WAIT=${4:-45}
DISP=${5:-:96}
HERE="$(cd "$(dirname "$0")" && pwd)"

module load hpctoolkit/2025.1.2 >/dev/null 2>&1
echo "hpcviewer: $(command -v hpcviewer)"

pkill -f "Xvfb $DISP" 2>/dev/null; sleep 1
Xvfb "$DISP" -screen 0 "${GEO}x24" -nolisten tcp >/tmp/xvfb_hv.log 2>&1 &
XVFB=$!
sleep 2
export DISPLAY="$DISP"

hpcviewer "$DB" >/tmp/hpcv.log 2>&1 &
HV=$!
echo "waiting ${WAIT}s for Eclipse startup + DB load..."
sleep "$WAIT"

# Optional XTEST clicks (switch to Trace view tab, pick a rank, etc.).
if [ -n "${HV_CLICKS:-}" ]; then
  python3 -m pip install --user --break-system-packages --quiet python-xlib 2>&1 | tail -1
  python3 "$HERE/xtest_click.py" ${HV_CLICKS} 2>&1 | tail -4
  sleep 3
fi

# Optional rubber-band drag to zoom a time range in the trace Main view.
if [ -n "${HV_DRAG:-}" ]; then
  python3 -m pip install --user --break-system-packages --quiet python-xlib 2>&1 | tail -1
  python3 "$HERE/xtest_drag.py" ${HV_DRAG} 2>&1 | tail -2
  sleep 3
fi

# Optional second rubber-band drag (deep zoom on the already-zoomed view).
if [ -n "${HV_DRAG2:-}" ]; then
  python3 "$HERE/xtest_drag.py" ${HV_DRAG2} 2>&1 | tail -2
  sleep 3
fi

# Optional clicks issued *after* the zoom (e.g. re-select a sample to refresh the
# call-stack pane at the new zoom level).
if [ -n "${HV_CLICKS2:-}" ]; then
  python3 "$HERE/xtest_click.py" ${HV_CLICKS2} 2>&1 | tail -4
  sleep 2
fi

# Optional keypress (e.g. 'asterisk' to expand a selected SWT tree subtree).
if [ -n "${HV_KEY:-}" ]; then
  python3 "$HERE/xtest_key.py" ${HV_KEY} 2>&1 | tail -2
  sleep 2
fi

# Park the pointer in the empty background so no tooltip/selection overlays the shot.
python3 - "$DISP" "${HV_PARK:-1800 600}" <<'PY' 2>/dev/null
import sys, time
from Xlib import X, display
from Xlib.ext import xtest
d = display.Display()
x, y = (int(v) for v in sys.argv[2].split())
xtest.fake_input(d, X.MotionNotify, x=x, y=y); d.sync(); time.sleep(0.6)
PY

xwd -root -display "$DISP" -silent -out /tmp/hpcv.xwd 2>/tmp/xwd_hv.log
python3 "$HERE/xwd2png.py" /tmp/hpcv.xwd "$OUT"

kill "$HV" 2>/dev/null; kill "$XVFB" 2>/dev/null
echo "--- hpcviewer log (tail) ---"; tail -8 /tmp/hpcv.log 2>/dev/null
