#!/usr/bin/env bash
# =============================================================================
# perf_security_demo.sh — demonstrate perf_events access control & security
# for the CG-CPU solver, following the kernel guide:
#   https://docs.kernel.org/admin-guide/perf-security.html
#
# It is READ-ONLY and needs NO root: it reports the current perf security
# posture (paranoid level, CAP_PERFMON, resource limits), explains what the
# active level allows, then runs the perf scopes that the current level permits
# against ./cg_cpu, marking each PASS / SKIP.
#
# perf security is a KERNEL feature and is independent of the ROCm version, so
# this demo pins ONE rocm module only to build cg_cpu; it does not sweep ROCm.
#
# Usage:
#   module load rocm/7.2.3 openmpi         # any one rocm; kernel-level demo
#   cd CG-CPU && make CXXFLAGS="-O3 -g -std=c++17"
#   ./perf_security_demo.sh [matrix] [seed]
# Run inside a Slurm allocation (login nodes are typically restricted):
#   salloc -p PPAC_MI300A_SPX --exclusive -t 00:20:00
# =============================================================================
set -uo pipefail

MATRIX="${1:-src/Dubcova2.pm}"
SEED="${2:-12345}"
APP=(./cg_cpu "$MATRIX" "$SEED")
PERF="${PERF:-perf}"

hr() { printf '%.0s-' {1..78}; echo; }
say() { printf '\n### %s\n' "$*"; }

command -v "$PERF" >/dev/null 2>&1 || { echo "ERROR: '$PERF' not found on PATH"; exit 1; }
[[ -x ./cg_cpu ]] || { echo "ERROR: build cg_cpu first (make CXXFLAGS=\"-O3 -g -std=c++17\")"; exit 1; }

# Launch the MPI binary via mpirun so a single rank initializes cleanly (a bare
# singleton can abort in the PML layer and suppress perf's counter print).
if command -v mpirun >/dev/null 2>&1; then LAUNCH=(mpirun -n 1 --oversubscribe); else LAUNCH=(); fi

# ---------------------------------------------------------------------------
say "1. Current perf_events security posture"
hr
PARANOID="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo '?')"
MLOCK="$(cat /proc/sys/kernel/perf_event_mlock_kb 2>/dev/null || echo '?')"
printf '%-28s = %s\n' "kernel"                 "$(uname -r)"
printf '%-28s = %s\n' "perf_event_paranoid"    "$PARANOID"
printf '%-28s = %s KiB (per-cpu perf mmap budget)\n' "perf_event_mlock_kb" "$MLOCK"
printf '%-28s = %s (RLIMIT_NOFILE; >= events*cpus)\n' "ulimit -n"  "$(ulimit -n)"
printf '%-28s = %s KiB (RLIMIT_MEMLOCK)\n' "ulimit -l"  "$(ulimit -l)"
# resolve the real ELF behind the Ubuntu /usr/bin/perf wrapper (caps live there)
PERF_ELF="/usr/lib/linux-tools/$(uname -r)/perf"; [[ -x "$PERF_ELF" ]] || PERF_ELF="$(command -v "$PERF")"
printf '%-28s = %s\n' "perf ELF"              "$PERF_ELF"
PELF_CAPS="$(getcap "$PERF_ELF" 2>/dev/null)"
printf '%-28s = %s\n' "getcap (perf ELF)"     "${PELF_CAPS:-<none>}"
printf '%-28s = %s\n' "id"                    "$(id -un) (euid=$(id -u))"
if id -nG 2>/dev/null | tr ' ' '\n' | grep -qx "perf_users"; then
  printf '%-28s = %s\n' "perf_users member"    "yes (least-privilege perf setup active)"
else
  printf '%-28s = %s\n' "perf_users member"    "no (using open perf_event_paranoid; see perf_users_setup.sh)"
fi

# ---------------------------------------------------------------------------
say "2. What the active perf_event_paranoid level allows (kernel guide)"
hr
cat <<'EOF'
 -1 : no scope restrictions; per-cpu mlock limit ignored (least secure)
 >=0: per-process AND system-wide; excludes raw/ftrace tracepoints
 >=1: per-process only (no system-wide -a); user+kernel events
 >=2: per-process, USER-space events only (no kernel-space sampling)
 CAP_PERFMON (or root) bypasses these scope checks entirely.
EOF
case "$PARANOID" in
  -1) echo " -> This node: FULL access (all scopes; typical on exclusive compute nodes).";;
  0)  echo " -> This node: per-process + system-wide, no raw tracepoints.";;
  1)  echo " -> This node: per-process only; system-wide (-a) will be denied.";;
  2)  echo " -> This node: per-process USER-space only; kernel sampling denied.";;
  *)  echo " -> This node: paranoid=$PARANOID (>2): most restrictive; expect denials without CAP_PERFMON.";;
esac

# ---------------------------------------------------------------------------
say "3. Per-process counting (perf stat) — the CG-CPU baseline"
hr
"${LAUNCH[@]}" "$PERF" stat -e cycles,instructions,cache-references,cache-misses \
      "${APP[@]}" >/tmp/perf_demo_stat.$$ 2>&1
if grep -qaE 'insn per cycle|cache-misses|cache-references' /tmp/perf_demo_stat.$$; then
  grep -aE 'cycles|instructions|cache-|insn per|elapsed' /tmp/perf_demo_stat.$$
  echo "[PASS] per-process perf stat"
else
  echo "[SKIP] per-process perf stat denied at paranoid=$PARANOID (needs <=2, or CAP_PERFMON)"
  grep -iaE 'access|permission|paranoid|denied' /tmp/perf_demo_stat.$$ | head -3
fi
rm -f /tmp/perf_demo_stat.$$

# ---------------------------------------------------------------------------
say "4. System-wide counting (perf stat -a) — requires paranoid <= 0 or CAP_PERFMON"
hr
if timeout 6 "$PERF" stat -a -e cycles sleep 1 >/tmp/perf_demo_sys.$$ 2>&1; then
  grep -aE 'cycles|elapsed' /tmp/perf_demo_sys.$$
  echo "[PASS] system-wide perf stat -a"
else
  echo "[SKIP] system-wide perf stat -a denied (paranoid=$PARANOID > 0 and no CAP_PERFMON)"
fi
rm -f /tmp/perf_demo_sys.$$

# ---------------------------------------------------------------------------
say "5. Admin recipe: a least-privilege perf_users group (NOT run here)"
hr
cat <<'EOF'
 The recommended posture (see docs/09-perf-security-demo.md) grants CAP_PERFMON to
 a perf_users group on ONE ROCm release's profilers + the perf ELF, and LEAVES
 perf_event_paranoid open so other ROCm releases keep working. Drive it with the
 companion admin script (dry-run by default):

   ./perf_users_setup.sh --users "alice bob" \
       --rocm /nfsapps/ubuntu-24.04/opt/rocm-7.2.3         # prints the plan
   sudo ./perf_users_setup.sh --apply --users "alice bob" \
       --rocm /nfsapps/ubuntu-24.04/opt/rocm-7.2.3         # executes (root)

 It runs, per target and filesystem: groupadd/usermod, chgrp+chmod o-rwx, and
 setcap 'cap_perfmon,cap_syslog=ep' where the FS supports file capabilities
 (xfs/ext4). ROCm on NFS can't hold file caps -> group members get CAP_PERFMON
 via the kernel guide's capsh privileged shell, or rely on the open paranoid.
EOF

# ---------------------------------------------------------------------------
say "6. Multi-rank note (perf_event_mlock_kb budget)"
hr
cat <<EOF
 perf_event_mlock_kb=$MLOCK KiB is a PER-CPU budget. The first perf process can
 grab it all, starving other ranks. For per-rank MPI profiling, cap each rank's
 ring buffer with --mmap-pages, e.g.:

   mpirun -n 4 bash -c 'perf stat -o perf_r\${OMPI_COMM_WORLD_RANK}.txt \\
     -e cycles,instructions,cache-misses ${APP[*]}'
   # or for sampling:  perf record --mmap-pages=64 ...

 CAP_IPC_LOCK (or a privileged perf_users setup) lifts this limit.
EOF
# ---------------------------------------------------------------------------
say "7. Prove the overlay (optional, needs sudo in an exclusive allocation)"
hr
cat <<EOF
 This script is read-only. To PROVE that the CAP_PERFMON overlay actually beats a
 tightened kernel, run the companion test once, under sudo, in an EXCLUSIVE node:

   sudo ./perf_paranoid_test.sh 2        # raises perf_event_paranoid to 2 (or 3)

 It saves the current paranoid, raises it, runs perf AS YOU (not root) to show a
 system-wide 'perf stat -a' still works via CAP_PERFMON, then restores paranoid.
EOF
echo; echo "Demo complete. perf security is kernel-level: same on every ROCm module."
