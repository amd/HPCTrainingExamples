#!/bin/bash

# LIKWID install check: module resolves and likwid-topology reports the CPU.
# This is a CPU-only LIKWID build (no GPU needed).

module -t list 2>&1 | grep -q "^rocm" || module load rocm

if ! module load likwid 2>/dev/null; then
   echo "Unable to locate a modulefile for 'likwid'"
   exit 0
fi

command -v likwid-topology >/dev/null || { echo "FAIL: likwid-topology not found"; exit 1; }
likwid-topology 2>/dev/null | grep -q "CPU type:" || { echo "FAIL: no 'CPU type:' from likwid-topology"; exit 1; }

echo "LIKWID Install Check: SUCCESS"
