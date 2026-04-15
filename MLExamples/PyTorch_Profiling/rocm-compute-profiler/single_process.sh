#!/usr/bin/env bash

# For the ROCm Compute Profiler, formerly Omniperf.

PROFILER_TOP_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"

# Call the software set up script:
source ${PROFILER_TOP_DIR}/setup.sh

export NPROCS=1

pushd ${PROFILER_TOP_DIR}
if [ ! -f data/cifar-100-python ]; then
   ./download-data.sh
fi
popd

# Execute the python script:
rocprof-compute profile --no-roof -b 2.1.2 2.1.3 2.1.4 2.1.5 --name cifar_100_single_proc -- \
${PROFILER_TOP_DIR}/no-profiling/single_process.sh

rocprof-compute analyze -p workloads/cifar_100_single_proc/MI* -b 2.1.2 2.1.3 2.1.4 2.1.5
