Originally from HIP-Examples as cuda-stream
   https://github.com/ROCm-Developer-Tools/HIP-Examples
Code based on the code developed by John D. McCalpin
http://www.cs.virginia.edu/stream/FTP/Code/stream.c

cuda version written by: Massimiliano Fatica, NVIDIA Corporation

Further modifications by: Ben Cumming, CSCS

Ported to HIP by: Peng Sun, AMD

The benchmark is modified from STREAM benchmark implementation with the following kernels:
```
    COPY:       a(i) = b(i)
    SCALE:      a(i) = q*b(i)
    SUM:        a(i) = b(i) + c(i)
    TRIAD:      a(i) = b(i) + q*c(i)
```

For ROCm environment
```
   module load rocm
   module load cmake
   export CXX=${ROCM_PATH}/llvm/bin/clang++
```

For ROCm with make
```
   make
```

For ROCm with cmake
```
   mkdir build && cd build
   cmake ..
   make VERBOSE=1
   ./stream
   ctest
```

For CUDA environment
```
   module load rocm
   module load CUDA/11.8
   module load cmake
```

For CUDA with make
```
   HIP_PLATFORM=nvidia make
```

For CUDA with cmake
```
   mkdir build && cd build
   cmake -DCMAKE_GPU_RUNTIME=CUDA ..
   make VERBOSE=1
   ./stream
   ctest
```
