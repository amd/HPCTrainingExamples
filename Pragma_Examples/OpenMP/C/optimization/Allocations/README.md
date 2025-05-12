# Investigate memory allocation problem: Dynamic memory allocations and memory pool on MI300A
This excercise highlights the importance of the reduction of dynamic memory allocations on MI300A 
with unified memory. So this excercise teaches you about a common performance problem on MI300A as 
well as some possible problems.

## version 1: Allocation Problem
The first version is an OpenMP offload example with three kernels in an iterative loop as an example of a "dwarf" of an application with an iterative solution or multiple time steps.

Set up the environment

```
module load amdclang
export HSA_XNACK=1
```

```
cd 1_alloc_problem
```
Compile the code with:
```
make
```
and run the example
```
./alloc_problem
```
The total execution time is roughly: 950 ms

## version 2: Solution no dynamic allocation and deallocation
In this version the allocations are moved outside the iteration loop.
```
cd 2_opt_allocation
```
```
make
```
```
./opt_allocation
```
Allocations (especially with system allocators, i.e. 'allocate') with unified memory are costly on MI300A. The first kernel will take longer since the memory is allocated at first touch. Subsequent iteration work on already allocated memory will be much faster. The deallocation is also only done once at the very end and we do not pay the high price of it in each iteration. 
The total execution time is roughly: 219 ms, so we have a speedup of more than a factor of 4 in this example. The gain improves the more iterations we do not have to do dynamic allocations.
We can learn from this that dynamic allocations and deallocations on MI300A should be avoided. 
But what if that is very hard to do in a true app which uses different temporary arrays in each subroutine?

## version 3: Solution memory pool
A possible solution is to use a memory pool. An example pool you can use in C/C++ and Fortran is the library Umpire (developed by LLNL).
```
cd  3_memorypool
```

First install umpire
```
export hip_DIR=${ROCM_PATH}
./umpire_setup.sh
```

Set the `UMPIRE_PATH` to the installation location:
```
export UMPIRE_PATH=$HOME/Umpire
```

Have a look how the memory pool is set up by looking at the memorypool.f90 file.

Compile and run with umpire:

```
make
./memorypool
```

The first two iterations when the memory pool still 'warms up' will have a longer first touch kernel. But later iterations will be fast.

The total execution time is roughly: 253 ms, so the speed up is a little less than moving the allocation outside, but also for the memory pool we only pay the price in the first 2 iterations until the library figured out the size of the pool. This is usually negligible if hundreds to thousands of iterations / time steps are run in production applications which do more iterations than this training example.

Conclusion: The usage of memory pools is recommended if you have frequent allocations and deallocations in your application and cannot move the allocations outside the loop!
