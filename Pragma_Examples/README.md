# OpenMP Offload Fortran examples
The Fortran OpenMP porting examples can be found in the README.md in the Fortran directory
```
cd OpenMP/Fortran
```
The following examples in this Readme focus on C/C++. Change to the above mentioned one for Fortran.

# OpenMP Offload Intro Examples (C/C++)
**NOTE**: these exercises have been tested on MI210 and MI300A accelerators using a container environment.
To see details on the container environment (such as operating system and modules available) please see `README.md` on [this](https://github.com/amd/HPCTrainingDock) repo.

## Checking out makefiles and compiler toolchain

Running the first OpenMP example: `Pragma_Examples/OpenMP/C/saxpy`

### Build with AMDClang compiler

```bash
module load amdclang
make clean 
make 
./saxpy 
```

Confirm running on GPU with  

```bash
export LIBOMPTARGET_KERNEL_TRACE=1 
./saxpy
```

   * confirms that we are running on the GPU and also gives us the register usage 
   * Also could use `AMD_LOG_LEVEL=[0|1|2|3|4]` or `LIBOMPTARGET_KERNEL_TRACE=2`


## OpenMP Offload -- The Basics

We start out with the OpenMP threaded code for the CPU. This code is in 
`~/HPCTrainingExamples/Pragma_Examples/OpenMP/Intro` in the saxpy_cpu.cpp file. This
is the code on slide 16. We first load the amdclang module which will set the CXX
environment variable. This variable will get picked up by the Makefile for the build.

```
module load amdclang
make saxpy_cpu
./saxpy_cpu
```

The next example, saxpy1, is from slide 18 where the first version of OpenMP offloading is
tried. In this code, there is no map clause. The compiler can figure out the arrays that need to be copied 
over and their sizes.

```
make saxpy1
./saxpy1
```
While running one of these codelets, it may be useful to watch the GPU usage. Here are two approaches.

- open another terminal and `ssh` to the AAC node you are working on, or 
- use the tmux command
  
- run `watch -n 0.5 rocm-smi` command line from that terminal to visualize GPU activities.

Note that the basic tmux survival commands are:
```
cntl+b \"  - splits the screen  
cntl+b (up arrow) - move to the upper session
cntl+b (down arrow) - move to lower session
exit - end tmux session
```

Next, run the codelet on your preferred GPU device if you have allocated more than 1 GPU. For example, to execute on GPU ID #2, set the following environment variable: `export ROCR_VISIBLE_DEVICES=2` then run the code.

Profile the codelet and then compare output by setting
```bash
export LIBOMPTARGET_KERNEL_TRACE=1
export LIBOMPTARGET_KERNEL_TRACE=2
```
Note:

rocminfo can be used to get target architecture information.

The compile line uses the specific GPU architecture type. It grabs it from the rocminfo
command with a little bit of string manipulation. 

Let's now add a map clause as shown in quotes on slide 18 -- map(tofrom:y[0:N])

```
make saxpy2
./saxpy2
```

A lot of the initial optimization of an OpenMP offloading port is to minimize the
data movement from host to device and back. What is the optimum mapping of data
for this example? See saxpy3.cpp for the optimal map clauses.

```
make saxpy3
./saxpy3
```

In the example we have been working with so far, the compiler can determine the sizes
and will move the data for you. Let's see what happens when we have a subroutine
with pointers where the compiler does not know the sizes.

```
make saxpy4
./saxpy4
```

Try removing the map clause -- the program will now fail when you are working on discrete GPUs or with HSA_XNACK=0 on MI300A.
```
export HSA_XNACK=0
./saxpy4
```
and 
```
export HSA_XNACK=1
./saxpy4
```


## Multilevel Parallelism

We have been running on the GPU, but with only one thread in serial. Let's start adding parallelism.
The first thing we can do is add `#pragma omp parallel for simd` before the loop to tell it to run in parallel.

```
make saxpy5
./saxpy5
```

We have told it to run the loop in parallel, but we haven't given it any hardware resources. To add more
compute units, we need to add the teams clause. Then to spread the work across the threads, we need the 
distribute clause. (This code is currently not working ...)

```
make saxpy6
./saxpy6
```

More commonly, we add the triplet of `target teams distribute` to the pragma to enable all hardware
elements to the computation.

```
make saxpy7
./saxpy7
```

And in Fortran.

```
make saxpy2f
./saxpy2f
```

## Structured and Unstructured Target Data Regions

This example from slide 29 shows the use of a structured block region that encompasses several
compute loops. The data region persists across all of them, eliminating the need for map
clauses and data transfers.

```
make target_data_structured
./target_data_structured
```

This example shows the use of the target data to map the data to the device and then updating
it with the target update in the middle of the target data block.

```
make target_data_unstructured
./target_data_unstructured
```

When using larger data regions, it can be necessary to move data in the middle of the region
to support MPI communication or I/O. This example shows the use of the update clause to copy
new input from the host to the device.

```
make target_data_update
./target_data_update
```

# Advanced OpenMP Presentation

Here, we will discuss some examples that show more advanced OpenMP features.

## Memory Pragmas
First, we will consider the examples in the `CXX/memory_pragmas` directory:

```bash
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/CXX/memory_pragmas
```

### Exercises Setup

Setup your environment:

```bash
export LIBOMPTARGET_INFO=-1
export OMP_TARGET_OFFLOAD=MANDATORY
```
The first flag above will allow you to see OpenMP activity, while the second terminates the program if code fails to be executed on device (as opposed to falling back on the host). You can also be more selective in the output generated by using the individual bit masks:

```bash
export LIBOMPTARGET_INFO=$((0x01 | 0x02 | 0x04 | 0x08 | 0x10 | 0x20))
```

Create a build directory and compile using `cmake`: this will place all executables in the `build` directory:

```bash
mkdir build && cd build
cmake ..
make 
```
### Mem1 (Initial Version)

There are 12 versions of an initial example code called `mem1.cc`, which is an implementation of a `daxpy` kernel with a single pragma with a map clause at the computational loop:

```bash
void daxpy(int n, double a, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
#pragma omp target teams distribute parallel for simd map(to: x[0:n], y[0:n]) map(from: z[0:n])
        for (int i = 0; i < n; i++)
                z[i] = a*x[i] + y[i];
}

```
Run `mem1` to have an idea of what output is produced by the `LIBOMPTARGET_INFO=-1` flag, which should include OpenMP calls like the following:

```bash
Libomptarget device 0 info: Entering OpenMP kernel at mem1.cc:89:1 with 5 arguments:
Libomptarget device 0 info: firstprivate(n)[4] (implicit)
Libomptarget device 0 info: from(z[0:n])[80000]
Libomptarget device 0 info: firstprivate(a)[8] (implicit)
Libomptarget device 0 info: to(x[0:n])[80000]
Libomptarget device 0 info: to(y[0:n])[80000]
Libomptarget device 0 info: Creating new map entry with HstPtrBase=0x0000000001772200, ... 
Libomptarget device 0 info: Creating new map entry with HstPtrBase=0x000000000174b0e0, ... 
Libomptarget device 0 info: Copying data from host to device, HstPtr=0x000000000174b0e0, ...
Libomptarget device 0 info: Creating new map entry with HstPtrBase=0x000000000175e970, ...
Libomptarget device 0 info: Copying data from host to device, HstPtr=0x000000000175e970, ...
Libomptarget device 0 info: Mapping exists with HstPtrBegin=0x0000000001772200, ...
Libomptarget device 0 info: Mapping exists with HstPtrBegin=0x000000000174b0e0, ...
Libomptarget device 0 info: Mapping exists with HstPtrBegin=0x000000000175e970, ...
Libomptarget device 0 info: Mapping exists with HstPtrBegin=0x000000000175e970, ...
Libomptarget device 0 info: Mapping exists with HstPtrBegin=0x000000000174b0e0, ...
Libomptarget device 0 info: Mapping exists with HstPtrBegin=0x0000000001772200, ...
Libomptarget device 0 info: Copying data from device to host, TgtPtr=0x00007f617c420000, ...
Libomptarget device 0 info: Removing map entry with HstPtrBegin=0x000000000175e970, ...
Libomptarget device 0 info: Removing map entry with HstPtrBegin=0x000000000174b0e0, ...
Libomptarget device 0 info: Removing map entry with HstPtrBegin=0x0000000001772200, ...
-Timing in Seconds: min=0.010115, max=0.010115, avg=0.010115
-Overall time is 0.010505
Last Value: z[9999]=7.000000
```

Not all versions are discussed in this document. Using `vimdiff` to compare versions is useful to explore the differences, e.g.:

```bash
vimdiff mem1.cc mem2.cc
```

### Mem2 (Add enter/exit data alloc/delete when memory is created/freed)

The initial code in `mem1.cc` is modified to obtain `mem2.cc` with the following additions:

```bash
#pragma omp target enter data map(alloc: x[0:n], y[0:n], z[0:n]) // line 52
```

```bash
#pragma omp target exit data map(delete: x[0:n], y[0:n], z[0:n]) // line 82
```


### Mem3 (Replace map to/from with updates to bypass unneeded device memory check)

In `mem3.cc`, in addition to the changes in `mem2.cc`,  the `daxpy` kernel is modified as follows:

```bash
void daxpy(int n, double a, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
#pragma omp target update to (x[0:n], y[0:n])
#pragma omp target teams distribute parallel for simd
        for (int i = 0; i < n; i++)
                z[i] = a*x[i] + y[i];
#pragma omp target update from (z[0:n])
}
```

### Mem4 (Replace delete with release to use reference counting)

Compared to `mem2.cc`, `mem4.cc` differs only at line 82, where a delete is replaced with a release:

```bash
#pragma omp target exit data map(release: x[0:n], y[0:n], z[0:n]) // line 82
```

### Mem5 (Use enter data map to/from alloc/delete to reduce memory copies)

Similar to `mem2.cc`. this version differs from the original only at lines 52 and 82:

```bash
#pragma omp target enter data map(to: x[0:n], y[0:n]) map(alloc: z[0:n]) // line 52
```

```bash
#pragma omp target exit data map(from: z[0:n]) map(delete: x[0:n], y[0:n]) // line 82
```
### Mem7 (Use managed memory to automatically move data)

In this example, we epxloit automatic memory management by the operating system. To enable it, export:

```bash
export HSA_XNACK=1
```
We also need to include the following pragma:

```bash
#pragma omp requires unified_shared_memory // line 22
```

### Mem8 (Use unified shared memory with maps for backward compatibility)

Compared to `mem7.cc`, `mem8.cc` supports backward compatibility using maps and also:

```bash
#ifndef NO_UNIFIED_SHARED_MEMORY
#pragma omp requires unified_shared_memory
#endif
```
### Mem12 (Only runs on MI300A)

This example uses the APU programming model of MI300A and  unified addresses in OpenMP.

## Kernel Pragmas

This set of exercises is in: `HPCTrainingExamples/Pragma_Examples/OpenMP/CXX/kernel_pragmas`.

### Exercises Setup

You should unset the `LIBOMPTARGET_INFO` environment flag if previously set.

```bash
unset LIBOMPTARGET_INFO
```

Then, set these environment variable

```bash
export CXX=amdclang++
export LIBOMPTARGET_KERNEL_TRACE=1
export OMP_TARGET_OFFLOAD=MANDATORY
export HSA_XNACK=1
```

### Brief Exercises Description

The example `kernel1.cc` is the same as `memory_pragmas/mem11.cc` except for the pragma line below (from `kernel1.cc`):

```bash
cout << "-Overall time is " << main_timer << endl;
#pragma omp target update from(z[0])
```

The example `kernel2.cc` differs from `kernel1.cc` as it adds `num_threads(64)` to the pragma line in the `daxpy` kernel:

```bash
void daxpy(int n, double a, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
#pragma omp target teams distribute parallel for simd num_threads(64)
        for (int i = 0; i < n; i++)
                z[i] = a*x[i] + y[i];
}
```

Similarly,  example `kernel3.cc` differs from `kernel1.cc` as it adds `num_threads(64) thread_limit(64)` to the pragma line in the `daxpy` kernel:

```bash
void daxpy(int n, double a, double *__restrict__ x, double *__restrict__ y, double *__restrict__ z)
{
#pragma omp target teams distribute parallel for simd num_threads(64) thread_limit(64)
        for (int i = 0; i < n; i++)
                z[i] = a*x[i] + y[i];
}
```
Something to test On your own: uncomment line 15 in CMakeLists.txt (the one with -faligned-allocation -fnew-alignment=256).

Another option to explore is adding the attribute (std::align_val_t(128) ) to each new line, for example:

```bash
double *x = new (std::align_val_t(128) ) double[n];
```

# Real-World OpenMP Language Constructs

For all excercises in this section:
```bash
module load amdclang
git clone https://github.com/AMD/HPCTrainingExamples
```
either choose
```bash
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
```
or 
```bash
cd ~/HPCTrainingExamples/Pragma_Examples/OpenMP/C
```

***Note***: make sure the compilers are set to your preference. This can be obtained  by exporting the `FC` and `CC` environment variables:

```
export FC=<my favorite Fortran compiler>
export CC=<my favorite C compiler>
```
It is suggested for those that want to truly experience the effort, that you take all the 
pragma statements out of these examples and do the port yourself.

## Simple Reduction

The first example is a simple reduction:
```bash
cd reduction_scalar
make
./reduction_scalar
```
Now try the array form
```bash
cd ../reduction_array
make
./reduction_array
```

If your compiler passes, it supports at least simple array reduction clauses

## Device Routine

Subroutines called from within a target region also cause some difficulties. We must tell the compiler that we want
these compiled for the GPU. Note that device routines are not (yet) supported by all compilers!

For this example

```bash
cd ../device_routine
```
there are multiple versions to choose from in Fortran, either with an interface and an external routine or using a module. Hence one first needs to enter the selected subfolder, and then:

```bash
make
./device_routine
```

## Device Routine with Global Data

Including the use of data from global scope in device routines also causes difficulties. We have examples for
both statically sized arrays and dynamically allocated global data. 
Note that device routines are not (yet) supported by all compilers!
Also, this excercise only exists in the C version at the moment.

<!-- Johanna: I maybe will do a Fortran version and hopefully remember to update here... -->

```bash
cd ../device_routine_wglobaldata
make
./device_routine
```

```bash
cd ../device_routine_wdynglobaldata
make
./device_routine
```
