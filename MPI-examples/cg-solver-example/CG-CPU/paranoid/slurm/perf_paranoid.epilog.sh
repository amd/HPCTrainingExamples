#!/usr/bin/env bash
# =============================================================================
# SLURM Epilog drop-in — restore perf_event_paranoid after a job that used
# --comment=paranoid[=N]. Runs as ROOT on each compute node at job end.
#
# Install (as root), matching the prolog:
#   Epilog=/etc/slurm/perf_paranoid.epilog.sh   (or drop into an epilog.d/)
#
# Restore precedence:
#   1) the exact value stashed by the prolog (preferred), else
#   2) $PERF_PARANOID_DEFAULT (site default; -1 for these exclusive compute nodes).
# Jobs that never set a paranoid comment are left untouched.
# =============================================================================
set -uo pipefail

SYSCTL=/proc/sys/kernel/perf_event_paranoid
STASH_DIR="${PERF_PARANOID_STASH_DIR:-/run}"
STASH="$STASH_DIR/perf_paranoid.${SLURM_JOB_ID:-unknown}"
SITE_DEFAULT="${PERF_PARANOID_DEFAULT:--1}"     # open, matches exclusive compute nodes
TAG="perf_paranoid_epilog"
log() { logger -t "$TAG" -- "$*" 2>/dev/null || echo "$TAG: $*" >&2; }

if [[ -f "$STASH" ]]; then
  restore="$(cat "$STASH" 2>/dev/null)"
  rm -f "$STASH" 2>/dev/null
  [[ -z "$restore" ]] && restore="$SITE_DEFAULT"
else
  # No stash: only reset if this job actually requested paranoid (comment fallback).
  comment="${SLURM_JOB_COMMENT:-}"
  if [[ -z "$comment" && -n "${SLURM_JOB_ID:-}" ]] && command -v scontrol >/dev/null; then
    comment="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | grep -oP 'Comment=\K\S+' | head -1)"
  fi
  [[ "$comment" =~ (^|[,[:space:]])paranoid(=?(-?[0-9]+))?([,[:space:]]|$) ]] || exit 0
  restore="$SITE_DEFAULT"
fi

if printf '%s\n' "$restore" > "$SYSCTL" 2>/dev/null; then
  log "job ${SLURM_JOB_ID:-?}: perf_event_paranoid restored -> $restore"
else
  log "job ${SLURM_JOB_ID:-?}: FAILED to restore perf_event_paranoid=$restore"
fi
exit 0
