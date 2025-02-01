#!/bin/bash -l

# Call the software set up script:
source setup.sh

# to be updated:
export MASTER_ADDR=`hostname`
export MASTER_PORT=1234

# Run the workload only downloading the data:
python3 train_cifar_100.py --download-only
