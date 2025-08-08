#!/bin/bash
module load rocprof-tracedecoder

cd ../HIP/vectorAdd

#Edit the Makefile and add -g to the HIPFlags
make vectoradd
./vectoradd
rocprofv3 --att -d tracedecoder_vectorAdd -- ./vectoradd
tar -cvf ../../tracedecoder.tar tracedecoder_vectorAdd
make clean
rm -rf tracedecoder_vectorAdd
cd ../saxpy
make saxpy
./saxpy
rocprofv3 --att -d tracedecoder_saxpy -- ./saxpy
tar -Avf ../../tracedecoder.tar
make clean
rm -rf tracedecoder_saxpy
