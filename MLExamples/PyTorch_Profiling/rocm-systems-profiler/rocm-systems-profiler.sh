#!/usr/bin/env bash

# This script launches the python executable with fixed arguments:

# to be updated and overruled in job launcher scripts, but 
# in case they aren't set, declare these here:

if [[ -z "${MASTER_ADDR}" ]]; then
    export MASTER_ADDR=`hostname`
fi

if [[ -z "${MASTER_PORT}" ]];
then
    export MASTER_PORT=1234
fi

if [[ -z "${NPROCS}" ]];
then
    export NPROCS=1
fi

PROFILER_TOP_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# Call the software set up script:
source ${PROFILER_TOP_DIR}/setup.sh

pushd ${PROFILER_TOP_DIR}
if [ ! -d data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Remove any stale rocprofsys-python3-output directories from previous runs so
# that the glob in the final cd below (and any CTest pass regex that looks for
# "perfetto-trace-") cannot match leftovers from earlier invocations.
rm -rf ./rocprofsys-python3-output

# Create the configuration for the system profiler:
export RSP_CFG=${PROFILER_TOP_DIR}/rocm-system-profiler/rocprofiler-systems_$$.cfg
rocprof-sys-avail -G $RSP_CFG

# Execute the python script.

# NOTE on --num-workers: on the reference rocm/therock-23.1.0 stack (rocprof-sys
# v1.5.0) using 2 DataLoader workers completes in ~90s versus ~112s with 0
# workers, a ~20% wall-time win with 2/2 stability across re-runs; the gain
# comes from overlapping CIFAR-100 preprocessing with the profiled GPU steps.
# On older stacks (rocm/7.2.x, rocprof-sys v1.3.0) the test is intermittently
# flaky (~1/3 success) regardless of whether --num-workers is 0 or 2 due to a
# known v1.3.0 issue around forked DataLoader workers; keeping it at 2 does
# not make things worse there and is strictly better on the current nightly.
# Keep this <= 2 to stay well below any historical v1.3.0 deadlock threshold.
# Other pytorch_profiling_*.sh tests are unaffected; the default in
# train_cifar_100.py is still 4.
rocprof-sys-sample -c $RSP_CFG -- \
python3 ${PROFILER_TOP_DIR}/train_cifar_100.py --batch-size 256 --max-steps 10 \
--num-workers 2 \
--data-path ${PROFILER_TOP_DIR}/data
rc=$?

rm -f $RSP_CFG

# Surface the perfetto trace from the newest output dir so that the
# CTest PASS_REGULAR_EXPRESSION "perfetto-trace-" can match.
latest="$(ls -1dt rocprofsys-python3-output/*/ 2>/dev/null | head -1)"
if [[ -n "$latest" ]]; then
    cd "$latest"
    ls
else
    echo "ERROR: rocprof-sys produced no rocprofsys-python3-output/<ts>/ directory" >&2
fi

exit $rc

