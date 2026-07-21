#!/usr/bin/env bash
# =============================================================================
# perf_paranoid_test.sh — PROVE the perf_users / CAP_PERFMON overlay under a
# TIGHTENED perf_event_paranoid.
#
# Two ways to raise paranoid:
#   (A) PREFERRED — let the SLURM prolog do it (no sudo for you):
#         salloc -p PPAC_MI300A_SPX --exclusive -t 00:20:00 --comment=paranoid2
#         module load rocm/7.2.3
#         ./perf_paranoid_test.sh                 # "verify" mode, runs as you
#       (needs slurm/perf_paranoid.{prolog,epilog}.sh installed by an admin)
#   (B) SELF-CONTAINED — raise/restore it yourself with sudo:
#         sudo ./perf_paranoid_test.sh 2          # "raise" mode; auto-restores
#       (perf runs as $SUDO_USER so success is due to CAP_PERFMON, not root)
#
# In both cases: profiling that still works at a raised paranoid is proof the
# CAP_PERFMON overlay (installed by perf_users_setup.sh) is effective.
# See docs/09-perf-security-demo.md.
# =============================================================================
set -uo pipefail

SYSCTL=/proc/sys/kernel/perf_event_paranoid
LEVEL="${1:-2}"                        # target level for raise mode

hr()  { printf '%.0s-' {1..78}; echo; }
say() { printf '\n### %s\n' "$*"; }

PERF_ELF="/usr/lib/linux-tools/$(uname -r)/perf"; [[ -x "$PERF_ELF" ]] || PERF_ELF="$(command -v perf)"
HAS_CAP=no; getcap "$PERF_ELF" 2>/dev/null | grep -q cap_perfmon && HAS_CAP=yes
in_group() { id -nG 2>/dev/null | tr ' ' '\n' | grep -qx perf_users; }

# --- pick mode -------------------------------------------------------------
if [[ $(id -u) -eq 0 ]]; then
  MODE=raise
  TEST_USER="${SUDO_USER:-}"
  if [[ -z "$TEST_USER" || "$TEST_USER" == root ]]; then
    echo "WARN: no unprivileged SUDO_USER; running perf as root proves nothing." >&2
    echo "      Use:  sudo ./perf_paranoid_test.sh $LEVEL" >&2
  fi
  as_user() { if [[ -n "$TEST_USER" && "$TEST_USER" != root ]]; then runuser -u "$TEST_USER" -- bash -lc "$1"; else bash -lc "$1"; fi; }
else
  MODE=verify
  TEST_USER="$USER"
  as_user() { bash -lc "$1"; }
fi

# --- shared probes (per-process + system-wide), run as the ordinary user ----
PP=FAIL; SW=FAIL
probes() {
  echo "-- per-process:  perf stat -e cycles,instructions sleep 1"
  if as_user "perf stat -e cycles,instructions sleep 1" >/tmp/pp_$$ 2>&1 && grep -qaE 'instructions|insn per cycle' /tmp/pp_$$; then
    grep -aE 'cycles|instructions|elapsed' /tmp/pp_$$ | sed 's/^/    /'; PP=PASS
  else
    echo "    DENIED"; grep -iaE 'access|permission|paranoid' /tmp/pp_$$ | head -2 | sed 's/^/    /'; PP=FAIL
  fi
  rm -f /tmp/pp_$$
  echo "-- system-wide:  perf stat -a -e cycles sleep 1"
  if as_user "perf stat -a -e cycles sleep 1" >/tmp/sw_$$ 2>&1 && grep -qaE 'cycles' /tmp/sw_$$; then
    grep -aE 'cycles|elapsed' /tmp/sw_$$ | sed 's/^/    /'; SW=PASS
  else
    echo "    DENIED"; grep -iaE 'access|permission|paranoid' /tmp/sw_$$ | head -2 | sed 's/^/    /'; SW=FAIL
  fi
  rm -f /tmp/sw_$$
}

# ===========================================================================
if [[ "$MODE" == verify ]]; then
  CUR="$(cat "$SYSCTL")"
  say "Verify mode (no root) — reads the CURRENT perf_event_paranoid"
  hr
  echo "test user            : $USER"
  echo "perf_event_paranoid  : $CUR"
  echo "perf ELF             : $PERF_ELF"
  echo "  CAP_PERFMON        : $HAS_CAP"
  echo "perf_users member    : $(in_group && echo yes || echo no)"
  [[ -n "${SLURM_JOB_ID:-}" ]] && echo "SLURM job / comment  : ${SLURM_JOB_ID} / '${SLURM_JOB_COMMENT:-}'"

  say "Profile as '$USER' at paranoid=$CUR"
  hr
  probes

  say "Verdict"
  hr
  echo "per-process=$PP  system-wide=$SW  (CAP_PERFMON=$HAS_CAP)"
  if (( CUR <= 0 )); then
    echo ">> paranoid is OPEN ($CUR): probes pass trivially, so this is NOT a test yet."
    echo "   Resubmit with the prolog hook to tighten it:"
    echo "     salloc -p PPAC_MI300A_SPX --exclusive -t 00:20:00 --comment=paranoid2"
    echo "   then re-run this script. (Or self-contained: sudo $0 2)"
  else
    if [[ "$PP" == PASS ]]; then
      echo ">> per-process CPU profiling WORKS at paranoid=$CUR (perf auto-scopes to :u)."
      echo "   That covers 'perf stat/record ./cg_cpu', and the ROCm GPU profilers"
      echo "   (rocprofv3, rocprof-compute) which don't use perf_events at all -- so"
      echo "   they are unaffected by perf_event_paranoid."
    else
      echo ">> even per-process profiling was DENIED at paranoid=$CUR."
    fi
    if [[ "$SW" == PASS ]]; then
      echo ">> system-wide 'perf stat -a' ALSO works via CAP_PERFMON=$HAS_CAP -> full overlay proven."
    else
      echo ">> system-wide 'perf stat -a' (uncore/IMC counters) is blocked. To allow it"
      echo "   WITHOUT lowering paranoid, apply the CAP_PERFMON overlay:"
      echo "     sudo ./perf_users_setup.sh --apply --users \"$USER\" --rocm \$ROCM_PATH"
    fi
  fi
  exit 0
fi

# --- raise mode (root, self-contained) -------------------------------------
[[ -z "${SLURM_JOB_ID:-}" ]] && echo "WARN: not in a Slurm allocation; only do this on an EXCLUSIVE node." >&2
ORIG="$(cat "$SYSCTL")"
restore() { echo "$ORIG" > "$SYSCTL" 2>/dev/null; echo "[restored] perf_event_paranoid = $(cat "$SYSCTL")"; }
trap restore EXIT INT TERM

say "1. Baseline"
hr
echo "test user            : ${TEST_USER:-root}"
echo "perf ELF             : $PERF_ELF   (CAP_PERFMON=$HAS_CAP)"
echo "perf_event_paranoid  : $ORIG (original)"

say "2. Raise perf_event_paranoid to $LEVEL (tighten security)"
hr
echo "$LEVEL" > "$SYSCTL"
echo "perf_event_paranoid  : $(cat "$SYSCTL") (raised)"

say "3. Profile as '${TEST_USER:-root}' under the tightened setting"
hr
probes

say "4. Verdict"
hr
echo "perf ELF has CAP_PERFMON : $HAS_CAP"
echo "per-process probe        : $PP"
echo "system-wide probe        : $SW"
if [[ "$SW" == PASS && "$HAS_CAP" == yes && -n "$TEST_USER" && "$TEST_USER" != root ]]; then
  echo
  echo ">> OVERLAY PROVEN: an ordinary perf_users member profiled the node at"
  echo "   perf_event_paranoid=$LEVEL because the perf binary carries CAP_PERFMON."
  echo "   Other ROCm releases (uncapped) still rely on the normally-open paranoid."
elif [[ "$SW" == FAIL ]]; then
  echo
  echo ">> Profiling was DENIED at paranoid=$LEVEL. Run the setup first:"
  echo "     sudo ./perf_users_setup.sh --apply --users \"$TEST_USER\" --rocm \$ROCM_PATH"
  echo "   (grants CAP_PERFMON to the perf ELF), then re-run this test."
fi
# paranoid restored by the EXIT trap
