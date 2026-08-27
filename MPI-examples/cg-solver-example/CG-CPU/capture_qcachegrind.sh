#!/bin/bash
# Headless screen capture of QCachegrind viewing a callgrind profile.
# Runs qcachegrind on a virtual X display (Xvfb) and grabs the window to PNG.
#   Usage: capture_qcachegrind.sh <callgrind.out> <out.png> [WxH] [wait_s]
set -u
CG=${1:?callgrind.out}; OUT=${2:?out.png}; GEO=${3:-1680x1050}; WAIT=${4:-9}
RES="${GEO}x24"

module purge 2>/dev/null
module load qcachegrind/23.08.5
echo "qcachegrind: $(command -v qcachegrind)"

# Seed the primary event type (e.g. D1mr = L1 data read miss) so the flat-profile
# columns show cache misses on load, with no interactive clicking required.
EVENT=${CG_EVENT:-Ir}; EVENT2=${CG_EVENT2:-Ir}
CONF="$HOME/.config/kde.org/QCachegrind.conf"; mkdir -p "$(dirname "$CONF")"
cat > "$CONF" <<EOF
[GeneralSettings]
RecentFiles=$CG

[Layouts]
Count=1
Current=0

[TracePositions]
EventType=$EVENT
EventType2=$EVENT2
EOF

export QT_QPA_PLATFORM=xcb
export DISPLAY=:99
pkill -f "Xvfb :99" 2>/dev/null; sleep 1
Xvfb :99 -screen 0 "$RES" -nolisten tcp >/tmp/xvfb.log 2>&1 &
XVFB=$!
sleep 2

qcachegrind -qwindowgeometry "${GEO}+0+0" "$CG" >/tmp/qcg.log 2>&1 &
QCG=$!
echo "waiting ${WAIT}s for qcachegrind to render..."
sleep "$WAIT"

# Optional: change the toolbar event-type combo via XTEST keyboard navigation.
#   CG_COMBO_XY="x y"  = pixel of the event-type combo;  CG_DOWN = ArrowDown count.
if [ -n "${CG_DOWN:-}" ]; then
  python3 -m pip install --user --break-system-packages --quiet python-xlib 2>&1 | tail -1
  set -- ${CG_COMBO_XY:-520 40}
  python3 "$(dirname "$0")/xtest_select.py" "$1" "$2" "$CG_DOWN" 2>&1 | tail -2
  sleep 2
fi

# Optional: extra left-clicks (e.g. select a function row, then a bottom tab).
if [ -n "${CG_CLICKS:-}" ]; then
  python3 -m pip install --user --break-system-packages --quiet python-xlib 2>&1 | tail -1
  python3 "$(dirname "$0")/xtest_click.py" ${CG_CLICKS} 2>&1 | tail -4
  sleep 4   # allow graphviz 'dot' to render the call-graph pane
fi

xwd -root -display :99 -silent -out /tmp/qcg_shot.xwd 2>/tmp/xwd.log
python3 "$(dirname "$0")/xwd2png.py" /tmp/qcg_shot.xwd "$OUT"

kill "$QCG" 2>/dev/null; kill "$XVFB" 2>/dev/null
echo "--- qcachegrind log (tail) ---"; tail -5 /tmp/qcg.log 2>/dev/null
