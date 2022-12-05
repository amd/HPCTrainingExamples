Originally from HIP-Examples as cuda-stream 
   https://github.com/ROCm-Developer-Tools/HIP-Examples
Code based on the code developed by John D. McCalpin
http://www.cs.virginia.edu/stream/FTP/Code/stream.c

cuda version written by: Massimiliano Fatica, NVIDIA Corporation

Further modifications by: Ben Cumming, CSCS

Ported to HIP by: Peng Sun, AMD

The benchmark is modified from STREAM benchmark implementation with the following kernels:
    COPY:       a(i) = b(i)
    SCALE:      a(i) = q*b(i)
    SUM:        a(i) = b(i) + c(i)
    TRIAD:      a(i) = b(i) + q*c(i)

To compile HIP version:
    make
To execute:
    ./stream

To compile on NV node, use Makefile.titan.
