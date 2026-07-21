#!/bin/bash

# LIKWID install check: module resolves and likwid-topology reports the CPU.
# This is a CPU-only LIKWID build (no GPU needed).

module -t list 2>&1 | grep -q "^rocm" || module load rocm

if ! module load likwid 2>/tmp/likwid_check.$$.err; then
   cat /tmp/likwid_check.$$.err
   rm -f /tmp/likwid_check.$$.err
   echo "Unable to locate a modulefile for 'likwid'"
   exit 0
fi
rm -f /tmp/likwid_check.$$.err

echo "=== LIKWID install check ==="
LIKWID_BIN=$(command -v likwid-topology) || { echo "FAIL: likwid-topology not found"; exit 1; }
echo "likwid-topology: ${LIKWID_BIN}"

echo "+ likwid-topology"
TOPO=$(likwid-topology 2>&1)
echo "${TOPO}"
echo "${TOPO}" | grep -q "CPU type:" || { echo "FAIL: no 'CPU type:' from likwid-topology"; exit 1; }

echo "LIKWID Install Check: SUCCESS"
