
To run the rocprofv3 trace decoder

# ROCprof Trace Decoder

The hands-on exercises will go through how to collect trace decoder output. For how to install
the ROCProf trace decoder on pre-ROCm 7.0 versions. See the instructions at 
https://github.com/amd/HPCTrainingDock in the HPCTrainingDock/tools/scripts/rocprofiler-sdk_setup.sh
script. This script will install the rocprofiler-sdk, aqlprofile and rocprof-trace-decoder packages
into a separate directory and then prepend those paths before the ROCm paths.

## Setting up environment

If the ROCProf trace decoder is installed with a module, load the appropriate module. With the ROCm
7.0 version, the ROCProf trace decoder will be integrated into ROCm software.

```
module load rocprofiler-sdk
  or
module load rocprof-tracedecoder
```

All of these exercises are from the AMD HPC Training Examples which can be retrieved with the following:

```
git clone https://github.com/amd/HPCTrainingExamples
```

The examples will be either in the HPCTrainingExamples/HIP or HPCTrainingExamples/rocprof-tracedecoder
directories.

## Basic test -- vectorAdd

```
cd HPCTrainingExamples/HIP/vectorAdd
```

```
make vectoradd
./vectoradd
rocprofv3 --att -d tracedecoder_vectorAdd -- ./vectoradd
```

Transfer the files in the `tracedecoder_vectorAdd` directory to your local machine and read them into ROCprof Compute Viewer

Cleaning up afterwards

```
make clean
rm -rf tracedecoder_vectorAdd
```

## Saxpy

```
cd HPCTrainingExamples/HIP/saxpy
```

```
make saxpy
./saxpy
rocprofv3 --att -d tracedecoder_saxpy -- ./saxpy
```

Transfer the files in `tracedecoder_saxpy` to your local machine and read them into ROCprof Compute Viewer

Cleaning up afterwards

```
make clean
rm -rf tracedecoder_saxpy
```

## Matrix multiply - hip version

```
cd HPCTrainingExamples/HIP/dgemm
```

```
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
bin/dgemm -m 8192 -n 8192 -k 8192 -i 3 -r 10 -d 0,1,2,3 -o dgemm.csv
rocprofv3 --att -d tracedecoder_dgemm_hip -- bin/dgemm -m 8192 -n 8192 -k 8192 -i 3 -r 10 -d 0,1,2,3 -o dgemm.csv
```

Transfer the files in `tracedecoder_dgemm_hip` to your local machine and read them into ROCprof Compute Viewer

Cleaning up afterwards

```
make clean
rm -rf tracedecoder_dgemm_hip
cd ..
rm -rf build
```

## Matrix multiply library test (DGEMM)

```
cd HPCTrainingExamples/rocprof-tracedecoder
```

```
make
rocprofv3 --att -att-perfcounters "SQ_INSTS_LDS SQ_INSTS_VMEM SQ_INSTS_VMEM_WR SQ_INSTS_VMEM_RD" -d tracedecoder_dgemm_library -- ./dgemm`

```

Transfer files in `tracedecoder_dgemm_library` to your local system

Cleaning up afterwards

``
make clean
rm -rf tracedecoder_dgemm_library
```

# ROCprofiler Compute Viewer

The trace decoder data can be viewed in a separate program called ROCprofiler Compute Viewer. There
are pre-built binaries for Microsoft Windows and source code that can be compiled for others systems.


