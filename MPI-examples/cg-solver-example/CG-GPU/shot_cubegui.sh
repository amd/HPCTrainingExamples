#!/bin/bash
# Headless CubeGUI screenshot via Xvfb (now that xvfb/libxcb-cursor0 are installed).
set -u
CUBEX="$(readlink -fm "${1:-scorep_cg_isend/profile.cubex}")"
OUT="$(readlink -fm "${2:-../docs/profilers/figs/cg_cubegui.png}")"
APP=~/containers/squashfs-root/AppRun
DISP=:99

[ -f "$CUBEX" ] || { echo "no cubex: $CUBEX"; exit 1; }
mkdir -p "$(dirname "$OUT")"

# clean any stale server
pkill -f "Xvfb $DISP" 2>/dev/null; sleep 1
Xvfb $DISP -screen 0 1680x1050x24 >/tmp/xvfb.log 2>&1 &
XVFB=$!; sleep 3
export DISPLAY=$DISP QT_QPA_PLATFORM=xcb

echo "[info] launching cube on $CUBEX"
"$APP" "$CUBEX" >/tmp/cube_gui.log 2>&1 &
CUBE=$!
# give it time to load the file and render the trees
sleep 25

if ! kill -0 $CUBE 2>/dev/null; then
   echo "[FAIL] cube exited early:"; tail -20 /tmp/cube_gui.log
   kill $XVFB 2>/dev/null; exit 1
fi

echo "[info] capturing display"
/usr/bin/python3 - "$OUT" <<'PY'
import sys
from PIL import ImageGrab
img = ImageGrab.grab(xdisplay=":99")
img.save(sys.argv[1])
print("saved", sys.argv[1], img.size)
PY
RC=$?
# fallback via xwd if Pillow grab failed
if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
   echo "[info] Pillow grab failed; trying xwd"
   xwd -root -display $DISP -silent > /tmp/cube.xwd 2>/dev/null && \
     /usr/bin/python3 -c "from PIL import Image; Image.open('/tmp/cube.xwd').save('$OUT')" 2>/dev/null
fi

kill $CUBE 2>/dev/null; kill $XVFB 2>/dev/null
ls -la "$OUT" 2>/dev/null && echo "[ok] screenshot written" || echo "[FAIL] no screenshot"
