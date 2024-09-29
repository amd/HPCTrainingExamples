# HIP and OpenMP Interoperability

This hands-on exercise uses the code in HPCTrainingExamples/HIP-OpenMP/daxpy. We have code that uses both OpenMP and HIP. These require two separate passes with compilers: one with amdclang++ and the other with hipcc. Go to the directory containing the example and set up the environment:

```
cd HPCTrainingExamples/HIP-OpenMP/CXX/daxpy
module load rocm
export CXX=amdclang++
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

