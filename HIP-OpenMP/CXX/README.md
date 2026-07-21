# HIP and OpenMP Interoperability

README.md in `HPCTrainingExamples/HIP-OpenMP/CXX` from the Training Examples in repository.
If the `amdclang` is not available in your system, make sure to do `export CXX=amdclang++`.

## Full OpenMP Application Code 

The first example is just a straightforward openmp offload version
of saxpy. Any C++ compiler that supports OpenMP offload to hip should
work.

```
cd HPCTrainingExamples/HIP-OpenMP/CXX/saxpy_openmp_offload
module load rocm
module load amdclang
make
```

## OpenMP Application Calling a HIP Kernel

Now we move on to an OpenMP main calling a HIP version of the saxpy
kernel. Note that we have to get the device version of the array
pointers to pass into the HIP kernel, using `use_device_ptr(x,y)`.

```
cd HPCTrainingExamples/HIP-OpenMP/CXX/saxpy_openmp_hip
module load rocm
module load amdclang
unset HSA_XNACK
make
```

Try to create an equivalent version of the code in `saxpy_openmp.cc` that uses `omp target enter data` and `omp target exit data` instead of `omp target data`.

## APU Programming Model Version

With the APU programming model the explicit memory management handled with OpenMP in the `saxpy_open_hip` directory can now be removed. The code has to be run after setting `HSA_XNACK=1` to enable unified shared memory:

```
cd HPCTrainingExamples/HIP-OpenMP/CXX/saxpy_APU
module load rocm
module load amdclang
export HSA_XNACK=1
make
```

## HIP application calling an OpenMP Kernel
The next example does the converse of what we saw: it is a HIP application code calling an OpenMP kernel for saxpy executing on the GPU:

```
cd HPCTrainingExamples/HIP-OpenMP/CXX/saxpy_hip_openmp
module load rocm
module load amdclang
unset HSA_XNACK
make
```

Try to create a version that leverages the APU progamming model, in a similar way done for the OpenMP application calling a HIP kernel.

## OpenMP and HIP Kernels in the Same Source File

You can put both OpenMP and HIP code in the same source file with some care.
The next hands-on exercise shows how in the code in
`HPCTrainingExamples/HIP-OpenMP/daxpy`. We have code that uses both OpenMP
and HIP. These require two separate passes with compilers: one with
amdclang++ and the other with hipcc. Go to the directory containing the
example and set up the environment:

```
cd HPCTrainingExamples/HIP-OpenMP/CXX/daxpy
module load rocm
module load amdclang
```
View the source code file daxpy.cc and note the two #ifdef blocks.

 The first one is __DEVICE_CODE__ that we want to compile with hipcc.

 The second is __HOST_CODE__ that we will use the C++ compiler to compile.

 All of the HIP calls and variables are in the first block. The second block contains the OpenMP pragmas.

 While we can use hipcc to compile standard C++ code, it will not work on code with OpenMP pragmas. The call to the HIP daxpy kernel occurs near the end of the host code block. We could split out these two code blocks into separate files, but this may be more intrusive with a code design.

Now we can take a look at the Makefile we use to compile the code in the single file. In the file, we create two object files for the executable to be dependent on.

 We then compile one with the CXX compiler with `-D__HOST_CODE__` defined.

 The second object file is compiled using hipcc and with `-D__DEVICE_CODE__` defined.

 This doesn't completely solve all the issues with separate translation units, but it does help workaround some code organization constraints.

Now on to building and running the example.

```
make
./daxpy
```

