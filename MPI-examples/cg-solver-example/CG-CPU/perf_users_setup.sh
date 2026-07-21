#!/usr/bin/env bash
# =============================================================================
# perf_users_setup.sh — least-privilege perf access for profiling, WITHOUT
# lowering perf_event_paranoid, following the kernel guide:
#   https://docs.kernel.org/admin-guide/perf-security.html
#
# Model (see docs/09-perf-security-demo.md):
#   1. Create a `perf_users` group and add the chosen users.
#   2. Gate the profiling executables of ONE ROCm release (plus the perf ELF) to
#      that group (chgrp + chmod o-rwx) and, where the filesystem supports file
#      capabilities, stamp CAP_PERFMON on them (setcap).
#   3. LEAVE /proc/sys/kernel/perf_event_paranoid untouched (open) so profilers
#      from OTHER ROCm releases keep working unchanged.
#
# This is an ADMIN action. It is DRY-RUN by default: it prints every command it
# would run and changes nothing. Re-run with --apply (as root) to execute.
#
# Usage:
#   ./perf_users_setup.sh [--apply] \
#        [--group perf_users] \
#        [--users "alice bob"] \
#        [--rocm /nfsapps/.../rocm-7.2.3] \
#        [--profilers "rocprofv3 rocprof-compute rocprof-sys-run rocprof-sys-sample"]
#
# Defaults: group=perf_users, users=$USER, rocm=$ROCM_PATH, a sane profiler list.
# =============================================================================
set -uo pipefail

GROUP="perf_users"
USERS="${USER:-$(id -un)}"
ROCM="${ROCM_PATH:-}"
PROFILERS="rocprofv3 rocprof-compute rocprof-sys-run rocprof-sys-sample rocprof-sys-instrument"
APPLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift;;
    --group) GROUP="$2"; shift 2;;
    --users) USERS="$2"; shift 2;;
    --rocm) ROCM="$2"; shift 2;;
    --profilers) PROFILERS="$2"; shift 2;;
    -h|--help) sed -n '2,30p' "$0"; exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

# --- helpers ---------------------------------------------------------------
run() {  # print, and execute only when --apply
  printf '    %s\n' "$*"
  if [[ $APPLY -eq 1 ]]; then eval "$@"; fi
}
note() { printf '  # %s\n' "$*"; }
hr()   { printf '%.0s-' {1..78}; echo; }

# real ELF behind the Ubuntu /usr/bin/perf wrapper (setcap must target the ELF)
perf_elf() {
  local real="/usr/lib/linux-tools/$(uname -r)/perf"
  [[ -x "$real" ]] && { echo "$real"; return; }
  command -v perf 2>/dev/null
}

# does the filesystem holding $1 support file capabilities (security.capability)?
# nfs / nosuid mounts do not; xfs/ext4/btrfs do.
cap_capable_fs() {
  local f="$1" fstype opts
  fstype="$(stat -f -c '%T' "$f" 2>/dev/null)"
  opts="$(findmnt -no OPTIONS -T "$f" 2>/dev/null)"
  case "$fstype" in
    nfs*|autofs|tmpfs|overlay*) return 1;;
  esac
  [[ ",$opts," == *",nosuid,"* ]] && return 1
  return 0
}

if [[ $APPLY -eq 1 && $(id -u) -ne 0 ]]; then
  echo "ERROR: --apply must run as root (use: sudo $0 --apply ...)" >&2
  exit 1
fi

[[ $APPLY -eq 1 ]] && MODE="APPLY (executing)" || MODE="DRY-RUN (printing only; re-run with --apply as root)"
echo "perf_users_setup.sh — $MODE"
echo "group=$GROUP  users='$USERS'  rocm='${ROCM:-<unset>}'"
hr

# --- 1. group + membership -------------------------------------------------
echo "1) Create group and add users"
run "groupadd -f '$GROUP'"
for u in $USERS; do run "usermod -aG '$GROUP' '$u'"; done
note "users must log out/in for new group membership to take effect"
echo

# --- 2. build the target list: perf ELF + ROCm-release profilers -----------
echo "2) Gate + capability-stamp the profiling executables"
declare -a TARGETS
PELF="$(perf_elf)"; [[ -n "${PELF:-}" ]] && TARGETS+=("$PELF")
if [[ -n "$ROCM" ]]; then
  for p in $PROFILERS; do
    f="$ROCM/bin/$p"
    [[ -e "$f" ]] && TARGETS+=("$f") || note "skip (not in this release): $f"
  done
else
  note "no --rocm given: only the perf ELF is targeted (pass --rocm to gate a release)"
fi
echo

CAP='cap_perfmon,cap_syslog=ep'   # least privilege: perf_events scope + kallsyms
for f in "${TARGETS[@]}"; do
  echo "  target: $f"
  run "chgrp '$GROUP' '$f'"
  run "chmod o-rwx '$f'"
  if cap_capable_fs "$f"; then
    run "setcap '$CAP' '$f'"
    run "getcap '$f'"
  else
    note "filesystem has no file-capability support (NFS/nosuid):"
    note "  cannot setcap here. Group members get CAP_PERFMON via the capsh"
    note "  privileged shell (see docs ch.9 / kernel guide), or rely on the"
    note "  open perf_event_paranoid for this release."
  fi
  echo
done

# --- 3. paranoid: leave it OPEN --------------------------------------------
echo "3) perf_event_paranoid — LEFT UNCHANGED on purpose"
P="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
note "current perf_event_paranoid = $P (NOT modified by this script)"
note "we do NOT tighten it: profilers from OTHER ROCm releases -- which are not"
note "capability-stamped -- keep working exactly as today (on the exclusive"
note "compute nodes it is -1 = open). The perf_users + CAP_PERFMON setup above is"
note "a least-privilege OVERLAY on the one release, not a global change."
echo
hr
echo "Done ($MODE). Verify from a user shell with: ./perf_security_demo.sh"
