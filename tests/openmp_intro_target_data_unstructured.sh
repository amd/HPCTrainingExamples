#!/bin/bash

module load amdclang
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Intro
make target_data_unstructured
./target_data_unstructured

make clean
