# AMD HPC Training Examples Repo

Welcome to AMD's HPC Training Examples Repo! 

Here you will find a variety of examples to showcase the capabilities of AMD's GPU software stack.
Please be aware that the repo is continuously updated to keep up with the most recent releases of the AMD software.

## Repository Structure

Please refer to this table of contents to locate the exercises you are interested in sorted by topic. 

1. [**HIP**](https://github.com/amd/HPCTrainingExamples/tree/main/HIP)
   1. ***HIP Functionality Checks***
      1. [`query_device`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/Stream_Overlap): checks that `hipMemGetInfo` works.
   2. ***Basic Examples***
      1. [`Stream_Overlap`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/Stream_Overlap): this example shows how to share the workload of a GPU offload compation using several overlapping streams. The result is an additional gain in terms of time of execution due to the additional parallelism provided by the overlapping streams. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIP/Stream_Overlap/README.md).
      2. [`dgemm`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/dgemm): a (d)GEMM application created as an exercise to showcase simple matrix-matrix multiplications on AMD GPUs. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/dgemm/README.md).
      3. [`basic_examples`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/basic_examples): a collection of introductory exercises such as device to host data transfer and basic GPU kernel implementation. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/basic_examples/README.md).
      4. [`hip_stream`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/hip-stream): modification of the STREAM benchmark for HIP. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIP/hip-stream/README.md).
      5. [`jacobi`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/jacobi): distributed Jacobi solver, using GPUs to perform the computation and MPI for halo exchanges. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIP/jacobi/README.md).
      6. [`matrix_addition`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/matrix_addition): example of a HIP kernel performing a matrix addition. 
      7. [`saxpy`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/saxpy): example of a HIP kernel performing a saxpy operation. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/saxpy/README.md).
      8. [`stencil_examples`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/stencil_examples): examples stencils operation with a HIP kernel, including the use of timers and asyncronous copies.
      9. [`vectorAdd`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/vectorAdd): example of a HIP kernel to perform a vector add. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/vectorAdd/README.md).
      10. [`vector_addition_examples`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/vector_addition_examples): another example of a HIP kernel to perform vector addition, including different versions such as one using shared memory, one with timers, and a CUDA one to try [`HIPIFY`](https://github.com/amd/HPCTrainingExamples/tree/main/HIPIFY) and [`hipifly`](https://github.com/amd/HPCTrainingExamples/tree/main/hipifly) tools on. The examples in this directory are not part of the HIP test suite.
   3. ***CUDA to HIP Porting***
      1. [`HIPIFY`](https://github.com/amd/HPCTrainingExamples/tree/main/HIPIFY): example to show how to port CUDA code to HIP with HIPIFY tools. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIPIFY/README.md).
      2. [`hipifly`](https://github.com/amd/HPCTrainingExamples/tree/main/hipifly): example to show how to port CUDA code to HIP with hipifly tools. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/hipifly/vector_add/README.md).
   4. [`HIP-Optimizations`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP-Optimizations): a daxpy HIP kernel is used to show how an initial version can be optimized to improve performance. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP-Optimizations/daxpy/README.md).
   5. [`HIPFort`](https://github.com/amd/HPCTrainingExamples/tree/main/HIPFort): two examples that show how to use the hipfort interface to call hipblas functions from Fortran.
   6. [`HIPStdPar`](https://github.com/amd/HPCTrainingExamples/tree/main/HIPStdPar): several examples showing C++ Std Parallelism on AMD GPUs. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/HIPStdPar/CXX/README.md).
   7. [`HIP-OpenMP`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP-OpenMP): example on HIP/OpenMP interoperability. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP-OpenMP/README.md)
2. [**MPI-examples**](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples)
   1. ***Benchmarks***: GPU aware benchmarks (`collective.cpp` and `pt2pt.cpp`) to assess the performance of the communication libraries. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/MPI-examples/README.md) [`Video of Presentation`](https://www.youtube.com/watch?v=77bS4B2Mvmo&list=PLB4tvLCynFjQq2rOIjEy39IDaugXcFPNa&index=2&t=0s).
   2. [***GhostExchange***](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange): slimmed down example of an actual physics application where the solution is initialized on a square domain discretized with a Cartesian grid, and then advanced in parallel using GPU aware MPI communications. **NOTE**: detailed [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign/README.md) files are provided here for the different versions of the `GhostExchange_ArrayAssign` code, that showcase how to use `Omnitrace` to profile this application.
3. [**ManagedMemory**](https://github.com/amd/HPCTrainingExamples/tree/main/ManagedMemory): programming model exercises, topics covered are APU programming model, OpenMP, performance protability frameworks (Kokkos and RAJA) and discrete GPU programming model. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/ManagedMemory/README.md).
4. [**MLExamples**](https://github.com/amd/HPCTrainingExamples/tree/main/MLExamples): a variation of PyTorch's MNIST example code and a smoke test for mpi4py using cupy. Instructions on how to run and test other ML frameworks are in the [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/MLExamples/README.md).
5. [**Occupancy**](https://github.com/amd/HPCTrainingExamples/tree/main/Occupancy): example on modifying thread occupancy, using several variants of a matrix vector multiplication leveraging shared memory and launch bounds.
6. [**OmniperfExamples**](https://github.com/amd/HPCTrainingExamples/tree/main/OmniperfExamples): several examples showing how to leverage Omniperf to perform kernel level optimization using HIP. **NOTE**: detailed READMEs are provided on each subdirectory. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/OmniperfExamples/README.md).[`Video of Presentation`](https://www.youtube.com/watch?v=8gg3aNUsR44&list=PLB4tvLCynFjQq2rOIjEy39IDaugXcFPNa&index=2&t=5080s).
7. [**Omniperf-OpenMP**](https://github.com/amd/HPCTrainingExamples/tree/main/Omniperf-OpenMP): example showing how to leverage Omniperf to perform kernel level optimization using Fortran and OpenMP. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/Omniperf-OpenMP/README.md).
8. [**Omnitrace**](https://github.com/amd/HPCTrainingExamples/tree/main/Omnitrace)
   1. ***Omnitrace on Jacobi***: Omnitrace used on the Jacobi solver example. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/Omnitrace/README.md). 
   2. ***Omnitrace by Example***: Omnitrace used on several versions of the Ghost Exchange example:
      1. ***OpenMP Version***: [`READMEs`](https://github.com/amd/HPCTrainingExamples/blob/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign) available for each of the different versions of the example code. [`Video of Presentation`](https://vimeo.com/951998260).
      2. ***HIP Version***:  [`READMEs`](https://github.com/amd/HPCTrainingExamples/blob/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP) available for each of the different versions of the example code.
9. [**Pragma_Examples**](https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples): OpenMP (in Fortran, C, and C++) and OpenACC examples. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/Pragma_Examples).
10. [**Speedup_Examples**](https://github.com/amd/HPCTrainingExamples/tree/main/Speedup_Examples): examples to show the speedup obtained going from a CPU to a GPU implementation. [`README`](https://github.com/amd/HPCTrainingExamples/blob/main/Speedup_Examples/rzf_training/README.md).
11. [**atomics_openmp**](https://github.com/amd/HPCTrainingExamples/tree/main/atomics_openmp): examples on atomic operations using OpenMP.
12. [**Kokkos**](https://github.com/amd/HPCTrainingExamples/tree/main/Kokkos): runs the Stream Triad example with a Kokkos implementation. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/Kokkos/README.md).
13. [**Rocgdb**](https://github.com/amd/HPCTrainingExamples/tree/main/Rocgdb): debugs the [`HPCTrainingExamples/HIP/saxpy`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/saxpy) example with Rocgdb.[`README`](https://github.com/amd/HPCTrainingExamples/tree/main/Rocgdb/README.md). [`Video of Presentation`](https://www.youtube.com/watch?v=8gg3aNUsR44&list=PLB4tvLCynFjQq2rOIjEy39IDaugXcFPNa&index=1&t=0s).
14. [**Rocprof**](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprof): uses Rocprof to profile [`HPCTrainingExamples/HIPIFY/mini-nbody/hip/`](https://github.com/amd/HPCTrainingExamples/tree/main/HIPIFY/mini-nbody/hip). [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprof/README.md).
15. [**Rocprofv3**](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3): uses Rocprofv3 to profile [`HPCTrainingExamples/HIP/jacobi/`](https://github.com/amd/HPCTrainingExamples/tree/main/HIP/jacobi). [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/README.md).
16. [**rocm-blog-codes**](https://github.com/amd/HPCTrainingExamples/tree/main/rocm-blogs-codes): this directory contains accompany source code examples for select HPC ROCm blogs found at [https://rocm.blogs.amd.com](https://rocm.blogs.amd.com). [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/rocm-blogs-codes/README.md).
17. [**login_info**](https://github.com/amd/HPCTrainingExamples/tree/main/login_info)
    1. [***AAC***](https://github.com/amd/HPCTrainingExamples/tree/main/login_info/AAC): instructions on how to log in to the AMD Accelerator Cloud (AAC) resource. [`README`](https://github.com/amd/HPCTrainingExamples/tree/main/login_info/AAC/README.md).
18. [**Doc**](https://github.com/amd/HPCTrainingExamples/tree/main/Doc): directory with LaTeX and PDF documents that contain some of the most relevant README files properly formatted for ease of reading. The PDF document is obtained building the LaTeX document.


## Run the Tests

Most of the exercises in this repo can be run as a test suite by doing:

```
git clone https://github.com/amd/HPCTrainingExamples && \
cd HPCTrainingExamples && \
cd tests && \
./runTests.sh
```
You can also run a subset of the whole test suite by specifying the subset you are interested in as an input to the `runTests.sh` script. For instance: `./runTests.sh --pytorch`. To see a full list of the possible subsets that can be run: `./runTests.sh --help`.

**NOTE**: tests can also be run manually from their respective directories, provided the necessary modules have been loaded and they have been compiled appropriately.

## Feedback
We welcome your feedback and contributions, feel free to use this repo to bring up any issues or submit pull requests.
The software made available here is released under the MIT license, more details can be found in [`LICENSE.md`](https://github.com/amd/HPCTrainingExamples/blob/main/LICENSE.md).
