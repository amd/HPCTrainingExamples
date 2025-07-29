#!/bin/bash

module load rocm

REPO_DIR="$(dirname "$(dirname "$(readlink -fm "$0")")")"
cd ${REPO_DIR}/HIP/vectorAdd

make vectoradd
./vectoradd
rocprofv3 --att -d out -- ./vectoradd
if [ -f out/ui_output_agent_*_dispatch_1/se0_sm0_sl0_wv0.json ]; then
  echo "Found a json output from the rocprofv3 compute viewer and trace decoder"
  ls -l out/ui_output_agent_*_dispatch_1/se0_sm0_sl0_wv0.json
  echo "PASSED"
fi
make clean
