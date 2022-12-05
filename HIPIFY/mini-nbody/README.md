Example taken from HIP-Examples/mini-nbody
   Used for demonstrating hipify tool
   HIP code additions: Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
Original CUDA source from https://github.com/harrism/mini-nbody. Mark Harris.
Discussion: Chapter 31. Fast N-Body Simulation with CUDA -- GPU Gems 3,
   Lars Nyland, Mark Harris, Jan Prins
================================
mini-nbody: A simple N-body Code
================================

A simple gravitational N-body simulation in less than 100 lines of C code, with CUDA optimizations.

Benchmarks
----------

There are 5 different benchmarks provided for CUDA and MIC platforms.

1. nbody-orig : the original, unoptimized simulation (also for CPU)
2. nbody-soa  : Conversion from array of structures (AOS) data layout to structure of arrays (SOA) data layout
3. nbody-flush : Flush denormals to zero (no code changes, just a command line option)
4. nbody-block : Cache blocking
5. nbody-unroll / nbody-align : platform specific final optimizations (loop unrolling in CUDA, and data alignment on MIC)

Files
-----

nbody.c : simple, unoptimized OpenMP C code
timer.h : simple cross-OS timing code

Each directory below includes scripts for building and running a "shmoo" of five successive optimizations of the code over a range of data sizes from 1024 to 524,288 bodies.

cuda/ : folder containing CUDA optimized versions of the original C code (in order of performance on Tesla K20c GPU)
  1. nbody-orig.cu   : a straight port of the code to CUDA (shmoo-cuda-nbody-orig.sh)
  2. nbody-soa.cu    : conversion to structure of arrays (SOA) data layout (shmoo-cuda-nbody-soa.sh)
  3. nbody-soa.cu + ftz : Enable flush denorms to zero (shmoo-cuda-nbody-ftz.sh)
  4. nbody-block.cu  : cache blocking in CUDA shared memory (shmoo-cuda-nbody-block.sh)
  5. nbody-unroll.cu : addition of "#pragma unroll" to inner loop (shmoo-cuda-nbody-unroll.sh)

HIP/ : folder containing HIP optimized versions of the original C code (in order of performance on FIJI NANO)
  1. nbody-orig.cpp   : a straight port of the code to HIP (HIP-nbody-orig.sh)
  2. nbody-soa.cpp    : conversion to structure of arrays (SOA) data layout (HIP-nbody-soa.sh)
  3. nbody-block.cu  : cache blocking in CUDA shared memory (shmoo-cuda-nbody-block.sh)



mic/  : folder containing Intel Xeon Phi (MIC) optimized versions of the original C code (in order of performance on Xeon Phi 7110P)
  1. ../nbody-orig.cu : original code (shmoo-mic-nbody-orig.sh)
  2. nbody-soa.c     : conversion to structure of arrays (SOA) data layout (shmoo-mic-nbody-soa.sh)
  3. nbody-soa.cu + ftz : Enable flush denorms to zero (shmoo-mic-nbody-ftz.sh)
  4. nbody-block.c   : cache blocking via loop splitting (shmoo-mic-nbody-block.sh)
  5. nbody-align.c   : aligned memory allocation and vector access (shmoo-mic-nbody-align.sh)

