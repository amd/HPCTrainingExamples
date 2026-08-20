#!/bin/bash
# Headless roc-optiq screenshot via Xvfb (proves the GUI renders under a virtual
# X display, i.e. it will render inside a TurboVNC / noVNC XFCE desktop too).
#
# Usage: ./shot_roc_optiq.sh [trace.db|trace.rpd|project.rpv] [out.png]
#   With a trace/project argument it is opened with -f and the tool renders the
#   loaded timeline (topology tree + tracks + event table); without one the empty
#   welcome window (menus/toolbar/panes) is captured.
#
# roc-optiq opens a fixed 1280x720 window, so the Xvfb screen is sized to match
# and the grab is auto-cropped to the rendered content (no black border).
#
# Interactive click-through is still done in a real VNC/noVNC/X11 desktop
# (man aac6_vnc / aac6_novnc / aac6_x11); this path is for scripted/CI shots.
set -u
TRACE="${1:-}"
OUT="$(readlink -fm "${2:-../docs/profilers/figs/cg_roc_optiq.png}")"
DISP=:99

# roc-optiq: prefer the module (module load rocm/7.2.4 roc-optiq), fall back to
# the install path if the module is not on PATH (e.g. stale Lmod cache).
if ! command -v roc-optiq >/dev/null 2>&1; then
  export PATH="/nfsapps/ubuntu-24.04/opt/rocmplus-7.2.4/roc-optiq-v0.5.0/bin:$PATH"
fi
command -v roc-optiq >/dev/null 2>&1 || { echo "roc-optiq not found"; exit 1; }
mkdir -p "$(dirname "$OUT")"

# clean any stale server; size the virtual screen to the app's 1280x720 window
pkill -f "Xvfb $DISP" 2>/dev/null; sleep 1
Xvfb $DISP -screen 0 1280x720x24 >/tmp/xvfb_optiq.log 2>&1 &
XVFB=$!; sleep 3
export DISPLAY=$DISP

# OpenGL backend renders with software Mesa (no GPU needed on a login node);
# imgui file dialog avoids a native (GTK) dialog that needs a portal service.
ARGS=(--backend opengl --file-dialog imgui)
[ -n "$TRACE" ] && ARGS+=(-f "$(readlink -fm "$TRACE")")
echo "[info] launching: roc-optiq ${ARGS[*]}"
roc-optiq "${ARGS[@]}" >/tmp/roc_optiq_gui.log 2>&1 &
APP=$!

# Wait for the render to settle: with a trace, block until the tracks are parsed
# ("Track meta data loaded"); otherwise just give the welcome window a moment.
if [ -n "$TRACE" ]; then
  for i in $(seq 1 60); do
    grep -q "Track meta data loaded" /tmp/roc_optiq_gui.log && break
    kill -0 $APP 2>/dev/null || break
    sleep 1
  done
  sleep 5   # let the timeline paint after the tracks are loaded
else
  sleep 12
fi

if ! kill -0 $APP 2>/dev/null; then
  echo "[FAIL] roc-optiq exited early:"; tail -20 /tmp/roc_optiq_gui.log
  kill $XVFB 2>/dev/null; exit 1
fi

echo "[info] capturing display $DISP"
/usr/bin/python3 - "$OUT" <<'PY'
import sys
from PIL import ImageGrab, ImageChops, Image
img = ImageGrab.grab(xdisplay=":99").convert("RGB")
# auto-crop the black border (the app window may be smaller than the screen)
bbox = ImageChops.difference(img, Image.new("RGB", img.size, (0, 0, 0))).getbbox()
if bbox:
    img = img.crop(bbox)
img.save(sys.argv[1])
print("saved", sys.argv[1], img.size)
PY
RC=$?
if [ $RC -ne 0 ] || [ ! -s "$OUT" ]; then
  echo "[info] Pillow grab failed; trying xwd"
  xwd -root -display $DISP -silent > /tmp/roc_optiq.xwd 2>/dev/null && \
    /usr/bin/python3 -c "from PIL import Image; Image.open('/tmp/roc_optiq.xwd').save('$OUT')" 2>/dev/null
fi

kill $APP 2>/dev/null; kill $XVFB 2>/dev/null
ls -la "$OUT" 2>/dev/null && echo "[ok] screenshot written" || echo "[FAIL] no screenshot"
