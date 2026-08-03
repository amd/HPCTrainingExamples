#!/usr/bin/env bash
# =============================================================================
# SLURM Prolog drop-in — honor  --comment=paranoid[=N]  to set perf_event_paranoid
# for the duration of a job. Runs as ROOT on each compute node at job start.
#
# Install (as root, via sudo), one of:
#   * point slurm.conf at it directly:      Prolog=/etc/slurm/perf_paranoid.prolog.sh
#   * or drop it into an existing prolog.d/ that your site already sources.
# Pair it with perf_paranoid.epilog.sh so the value is always restored.
#
# Intended for EXCLUSIVE nodes (e.g. PPAC_MI300A_SPX --exclusive): the change is
# node-wide but the node belongs to one job, and the epilog restores it.
#
# Usage from the user side (deployed form is 'paranoid<num>', no '='):
#   salloc -p PPAC_MI300A_SPX --exclusive -t 00:20:00 --comment=paranoid    # -> 2 (default)
#   salloc ... --comment=paranoid3           # stricter
#   salloc ... --comment=paranoid1           # per-process only
#   sbatch --comment=paranoid2 job.sh
# ('paranoid=N' and bare 'paranoid' are also accepted.)
# Jobs WITHOUT a paranoid comment are untouched (exit 0, no change).
# =============================================================================
set -uo pipefail

SYSCTL=/proc/sys/kernel/perf_event_paranoid
STASH_DIR="${PERF_PARANOID_STASH_DIR:-/run}"
STASH="$STASH_DIR/perf_paranoid.${SLURM_JOB_ID:-unknown}"
TAG="perf_paranoid_prolog"
log() { logger -t "$TAG" -- "$*" 2>/dev/null || echo "$TAG: $*" >&2; }

# --- read the job comment (env first, scontrol fallback) -------------------
comment="${SLURM_JOB_COMMENT:-}"
if [[ -z "$comment" && -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null; then
  comment="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'Comment=\K\S+' | head -1)"
fi

# --- only act when the comment asks for it ---------------------------------
# matches 'paranoid<N>' (deployed form), 'paranoid=<N>', or bare 'paranoid'
# anywhere in a comma/space-separated comment list.
if [[ "$comment" =~ (^|[,[:space:]])paranoid(=?(-?[0-9]+))?([,[:space:]]|$) ]]; then
  want="${BASH_REMATCH[3]:-2}"          # bare 'paranoid' => 2
else
  exit 0
fi

# --- validate against the documented scope ladder --------------------------
case "$want" in
  -1|0|1|2|3|4) ;;
  *) log "job ${SLURM_JOB_ID:-?}: invalid paranoid='$want' in comment; ignoring"; exit 0 ;;
esac

# Optional raise-only policy: refuse to LOWER security below the current value.
# Uncomment to forbid users making the node MORE permissive via --comment.
#cur_now="$(cat "$SYSCTL" 2>/dev/null || echo -1)"
#if (( want < cur_now )); then
#  log "job ${SLURM_JOB_ID:-?}: refusing to lower paranoid $cur_now -> $want"; exit 0
#fi

# --- stash the current value so the epilog restores it exactly -------------
cur="$(cat "$SYSCTL" 2>/dev/null || echo)"
mkdir -p "$STASH_DIR" 2>/dev/null
printf '%s\n' "$cur" > "$STASH" 2>/dev/null

if printf '%s\n' "$want" > "$SYSCTL" 2>/dev/null; then
  log "job ${SLURM_JOB_ID:-?} (${SLURM_JOB_USER:-?}): perf_event_paranoid ${cur:-?} -> $want"
else
  log "job ${SLURM_JOB_ID:-?}: FAILED to set perf_event_paranoid=$want"
fi
exit 0
