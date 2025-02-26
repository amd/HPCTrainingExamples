## Part 1: Fortran with interface blocks

Let's start with the device routine in a separate file with an interface.

```
cd device_routine_with_interface
```

there are six code versions in enumerated folders:

```
0_device_routine_portyourself
1_device_routine_wrong
2_device_routine_usm
3_device_routine_map
4_device_routine_link
5_device_routine_enter_data
```

Starting with the CPU version to try and porting yourself

```
cd 0_device_routine_portyourself
```

Build and run

```
make
./device_routine
```

The result should be

```
Result: sum of x is 1000.000000000000
```

Now add the directive to the three loops in `device_compute.f90`

```
!$omp target teams distribute parallel do
```

For the last loop, it is also necessary to add `reduction(+:sum)`

This has been done for you in the `1_device_routine_wrong` directory

```
cd ../1_device_routine_wrong
```

Build the code

```
make
```

You should see an error.  

```
ld.lld: error: undefined symbol: compute_
```

The compute routine is created only for the host and not for the device. So we need to add the device target 
directive to the compute subroutine definition in `compute.f90`.

Moving to the next version at 2_device_routine_usm directory where the device target directive
has been added. 

```
cd ../2_device_routine_usm
```

Note the additions. In compute.f90:

```
      subroutine compute(x)
          implicit none
          !$omp requires unified_shared_memory
          !$omp declare target
```

and in device_compute.f90

```
 program device routine
...
         implicit none
         !$omp requires unified_shared_memory

...                                                                                                                                                         !$omp target teams distribute parallel do                                                                                                                                                       do .... 
```

Now build and run the example

```
make
./device_routine
```

For the case where we want to do explicit memory movement, we use maps as show in `03_device_routine_map`.

```
cd ../03_device_routine_map
```

We take out the `!$omp requires unified_shared_memory` and add `map(tofrom:x)` and `map(to:x)` clauses. We can run this
example as before:

```
make
./device_routine
```

Some of the other clauses that can be uses are the `device_type(nohost)` that only generates device code and `link(compute)` that
specifies the link for the declare target clauses. Check out the example at

```
cd ../4_device_routine_link
make
./device_routine
```

The last example shows the use of the enter/exit data directives. This is an example of the use of unstructured data movement
directives.

```
!$omp target enter data map(alloc:x(1:N))
!$omp target exit data map(delete:x)
```

These are added to the code in `5_device_routine_enter_data`

```
cd ../5_device_routine_enter_data
make
./device_routine
```

## Part 2: Fortran with modules


There are three versions

```
0_device_routine_with_module_portyourself
1_device_routine_with_module
2_device_routine_with_module_usm
```

We first check out the original code in `0_device_routine_with_module_portyourself`

```
cd 0_device_routine_with_module_portyourself
```

Build and run

```
make
./device_routine
make clean
```

Now try and add the directives to port the example code to run on the device (GPU).

The solution for explicit data movement using unstructured memory directives is in `1_device_routine_with_module`

```
cd ../1_device_routine_with_module
make
./device_routine
```

Examining the two source files, we see that we first need to add the compute 
directives:

```
!$omp target teams distribute parallel do
!$omp target teams distribute parallel do reduction(+:sum)
```

In addition, we need the explicit memory movement directives 

```
!$omp target enter data map(alloc:x(1:N))
!$omp target exit data map(delete:x)
```

But that is not all we need to do. We also need to add `!$omp declare target` in compute.f90 to tell the compiler
to generate a device version of the compute subroutine.

The next example shows the unified shared memory version.

```
cd ../2_device_routine_with_module_usm
```

We need to add `!$omp requires unified_shared_memory` to both source code files since they both will have
OpenMP target directives. Now we just need to add the compute directives as above and also add the 
`!$omp declare target` directive inside the subroutine definition in computemod.f90.

Now build and run

```
make
./device_routine
```
