#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C/device_routine_wglobaldata
make
./device_routine

make clean
