#!/bin/bash
module load rocprof-tracedecoder

cd ../HIP/vectorAdd

#Edit the Makefile and add -g to the HIPFlags
make vectoradd
rocprofv3 --att -d tracedecoder_vectorAdd -- ./vectoradd
tar -cf ../../tracedecoder.tar tracedecoder_vectorAdd
make clean
rm -rf tracedecoder_vectorAdd

cd ../saxpy
make saxpy
rocprofv3 --att -d tracedecoder_saxpy -- ./saxpy
tar -rf ../../tracedecoder.tar tracedecoder_saxpy
make clean
rm -rf tracedecoder_saxpy

cd ../dgemm
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
rocprofv3 --att -d tracedecoder_dgemm_hip -- bin/dgemm -m 8192 -n 8192 -k 8192 -i 3 -r 10 -d 0,1,2,3 -o dgemm.csv
tar -rf ../../tracedecoder.tar tracedecoder_dgemm_hip
make clean
rm -rf tracedecoder_dgemm_hip
cd ..
rm -rf build

cd ../../rocprof-tracedecoder/dgemm
make
rocprofv3 --att -d tracedecoder_dgemm_library -- ./dgemm
tar -rf ../../tracedecoder.tar tracedecoder_dgemm_library
make clean
rm -rf tracedecoder_dgemm_library
