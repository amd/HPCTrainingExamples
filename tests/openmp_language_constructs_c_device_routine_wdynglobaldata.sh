#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/device_routine_wdynglobaldata
make
./device_routine

make clean
