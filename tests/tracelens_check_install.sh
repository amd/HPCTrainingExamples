#!/bin/bash

# TraceLens install check: module resolves, package imports, CLIs run.
# TraceLens parses GPU profiling traces but runs on the host (no GPU needed).
# stderr is silenced: pandas' optional deps emit a benign NumPy 1.x/2.x warning.

module -t list 2>&1 | grep -q "^rocm" || module load rocm

if ! module load tracelens 2>/dev/null; then
   echo "Unable to locate a modulefile for 'tracelens'"
   exit 0
fi

python3 -c "import TraceLens" 2>/dev/null || { echo "FAIL: import TraceLens"; exit 1; }
TraceLens_generate_perf_report_pytorch --help 2>/dev/null | grep -qi usage || { echo "FAIL: TraceLens_generate_perf_report_pytorch --help"; exit 1; }
xprof --help 2>/dev/null | grep -qi usage || { echo "FAIL: xprof --help"; exit 1; }

echo "TraceLens Install Check: SUCCESS"
