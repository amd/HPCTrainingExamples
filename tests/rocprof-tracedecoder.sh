#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/vectorAdd

make vectoradd
./vectoradd
rocprofv3 --att -d out -- ./vectoradd
numfiles=`ls -l out/ui_output_agent_*_dispatch_1 |wc -l`
if [[ "$numfiles" -gt 10 ]]; then
  echo "Found json output from the rocprofv3 compute viewer and trace decoder"
  ls -l out/ui_output_agent_*_dispatch_1/
fi
make clean
rm -rf out
