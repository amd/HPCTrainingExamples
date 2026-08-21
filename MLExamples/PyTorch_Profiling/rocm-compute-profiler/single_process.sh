#!/usr/bin/env bash

# For the ROCm Compute Profiler, formerly Omniperf.

PROFILER_TOP_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# Call the software set up script:
source ${PROFILER_TOP_DIR}/setup.sh

# Dependency precondition. `analyze` enforces the exact pins in its own
# requirements.txt and this run ends in `analyze`, so verify them before
# spending around 25 minutes profiling. Fail fast rather than time out.
_rpc_bin="$(command -v rocprof-compute 2>/dev/null)"
if [ -n "${_rpc_bin}" ]; then
  _rpc_req="$(dirname "$(dirname "${_rpc_bin}")")/libexec/rocprofiler-compute/requirements.txt"
  if [ -f "${_rpc_req}" ] && ! python3 - "${_rpc_req}" <<'PINS'
import sys
from importlib.metadata import version, PackageNotFoundError
bad = []
for raw in open(sys.argv[1]):
    raw = raw.split('#')[0].strip()
    if not raw or '==' not in raw:
        continue
    name, want = raw.split('==', 1)
    try:
        have = version(name)
    except PackageNotFoundError:
        bad.append('%s: absent (needs %s)' % (name, want)); continue
    if have != want:
        bad.append('%s: %s installed, needs %s' % (name, have, want))
for b in bad:
    print('  ' + b)
sys.exit(1 if bad else 0)
PINS
  then
    echo "rocprof-compute dependency pins are unsatisfied under $(python3 -V 2>&1)."
    echo "'analyze' cannot run, so skipping the profile run instead of timing out."
    echo "See ${_rpc_req} for the required versions."
    exit 1
  fi
fi

export NPROCS=1

pushd ${PROFILER_TOP_DIR}
if [ ! -d data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Execute the python script:
rm -rf workloads/cifar_100_single_proc
rocprof-compute profile --no-roof --set compute_thruput_flops --name cifar_100_single_proc -- \
${PROFILER_TOP_DIR}/no-profiling/single_process.sh

rocprof-compute analyze -p workloads/cifar_100_single_proc/MI* -b 2.1.2 2.1.3 2.1.4 2.1.5
