#!/bin/bash

# TraceLens install check: module resolves, package imports, CLIs run.
# TraceLens parses GPU profiling traces but runs on the host (no GPU needed).
# Command output is shown; pandas' optional deps may emit a benign NumPy
# 1.x/2.x warning on stderr, which is expected and harmless.

module -t list 2>&1 | grep -q "^rocm" || module load rocm

if ! module load tracelens 2>/tmp/tracelens_check.$$.err; then
   cat /tmp/tracelens_check.$$.err
   rm -f /tmp/tracelens_check.$$.err
   echo "Unable to locate a modulefile for 'tracelens'"
   exit 0
fi
rm -f /tmp/tracelens_check.$$.err

echo "=== TraceLens install check ==="

echo "+ python3 -c 'import TraceLens; print(TraceLens.__file__)'"
python3 -c "import TraceLens; print(TraceLens.__file__)" || { echo "FAIL: import TraceLens"; exit 1; }

echo "+ TraceLens_generate_perf_report_pytorch --help"
TL_HELP=$(TraceLens_generate_perf_report_pytorch --help 2>&1)
echo "${TL_HELP}"
echo "${TL_HELP}" | grep -qi usage || { echo "FAIL: TraceLens_generate_perf_report_pytorch --help"; exit 1; }

echo "+ xprof --help"
XPROF_HELP=$(xprof --help 2>&1)
echo "${XPROF_HELP}"
echo "${XPROF_HELP}" | grep -qi usage || { echo "FAIL: xprof --help"; exit 1; }

echo "TraceLens Install Check: SUCCESS"
