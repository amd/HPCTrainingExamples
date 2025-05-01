# C Code -- Porting device routine exercises

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/C/5_device_routines` in Training Examples repository

This exercise will show how to port kernels which call a subroutine or function.
Each version has a sub-folder with a 
- serial CPU version to port yourself and a
- solution for unified memory and
- a solution with map clauses.

Build and run analogous to the previous exercises.

There are three different versions:
```
cd 1_device_routine 
```
Explore the serial CPU code first. 

```
cd 0_device_routine_portyourself
module load amdclang
make
./device_routine
```
The rocm module will be loaded with the amdclang module. And the current rocm module will set `HSA_XNACK=1`. If there are no modules set up on your system, set the `CC` environment variable to the full path to the C compiler you want to use. Add the ROCm directory to the PATH and also the LLVM directory under ROCm. Also add the lib directory to the `LD_LIBRARY_PATH`. And finally set `HSA_XNACK` with `export HSA_XNACK=1`.

```
make
./device_routine
```
You should see the result:

```
Result: sum of x is 1000.000000
```

Now try and convert the example to run on the GPU. Start with adding `#pragma omp target teams distribute parallel for` before the for loops in the main program in `device_routine.c`. Note that one of the loops also needs a `reduction(+:sum)` clause added to the target directive. How do you show the compiler to compile the function in the other file, compute.c, for the GPU? Try adding the `#pragma omp declare target` directive to the subroutine declaration in compute.c.

There are two solutions for this exercise. One with the APU programming model using unified shared memory. The other has explicit map clauses for when unified shared memory is not available or not being used. We'll look at the unified shared memory version first.

```
cd ../1_device_routine_usm
```

Look at the two C source files and compare to the originals in `0_device_routine_portyourself`. To build and run the example:

```
make
./device_routine
```

Similarly with the solution using map clauses:

```
cd ../2_device_routine_map
```

Look for the map clauses in the `device_routine.c` source file. In this case, The memory is only accessed on the GPU. So, we use map(alloc:x[0:N]) and map(release:x[0:N]) in the clauses. Build and run the examples.

```
make
./device_routine
```

```
cd 2_device_routine_wglobaldata
```

First look at the original code in `0_device_routine_wglobaldata_portyourself`. 

```
cd 0_device_routine_wglobaldata_portyourself
```

Note the addition of the `global_data.c` file with the definition of the constants array. Build and run the example. 

```
make
./device_routine
```

Now try modifying the example to run on the GPU. How do you use the global data from the `global_data.c` file in your device subroutine?

For the solution, lets look at the example in `1_device_routine_wglobaldata`.

```
cd 1_device_routine_wglobaldata
```

Look at the directive `#pragma omp declare target` in the `global_data.c` file. Is this necessary for your version of the compiler?

It is a bit more complicated if the data being used is dynamically allocated. We have to be sure and map it over to the GPU after the memory allocation. We can experiment with this case in the next example.

```
cd ../3_device_routine_wdynglobaldata
```

Again there is a version that you can try and port before looking at the solution. 

```
cd 0_device_routine_wdynglobaldata_portyourself
```

Look at the `global_data.c` file and experiment with the right directive to move the data to the GPU.

The solution is also available.

```
cd 1_device_routine_wdynglobaldata
```

See the directives used to move the constants array to the GPU. Note that we also need to add declare target on the pointer to the array.


```
#pragma omp target enter data map(alloc:constants[0:isize])
```

In this example, we initialize the data on the GPU with:

```
#pragma omp target teams distribute parallel for
   for (int i = 0; i< isize; i++) {
      constants[i] = (double)i;
   }
```

How would this be different if we initialized the data on the CPU?

