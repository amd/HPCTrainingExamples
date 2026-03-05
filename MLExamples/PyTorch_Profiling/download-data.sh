#!/bin/bash

# Call the software set up script:

module list 2>&1 | grep -q -w "rocm"
if [ $? -eq 1 ]; then
  echo "modules are not loaded, loading them"
  source setup.sh
fi

# to be updated:
if [[ -z "${MASTER_ADDR}" ]]; then
    export MASTER_ADDR=`hostname`
fi

if [[ -z "${MASTER_PORT}" ]]; then
    export MASTER_PORT=1234
fi

# Run the workload only downloading the data:
python3 train_cifar_100.py --download-only
