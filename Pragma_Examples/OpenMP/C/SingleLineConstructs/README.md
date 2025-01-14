# OpenMP Single Line Compute Constructs:

We start with adding a single line directive to move the computation of a loop to the GPU. The exercises for this will utilize
the saxpy example.

## CPU version

This example uses OpenMP on the CPU with threading for parallelism. The pragma used is

```
#pragma omp parallel for
```

We go to the directory with the example and load the amdclang module. We can then build and run the code.

```
cd HPCTrainingExamples/Pragma_Examples/OpenMP/C/SingleLineConstructs
module load amdclang
make saxpy_cpu
./saxpy_cpu
```

You should get some output like:

```
Time of kernel: 0.188165
check output:
y[0] 4.000000
y[N-1] 4.000000
```

For the Fortran version, we just go to the corresponding directory and follow the same steps.

```
cd ../../Fortran/SingleLineConstructs
module load amdflang-new-beta-drop
make saxpy_cpu
./saxpy_cpu
```

The output

```
Time of kernel: 0.151135
 plausibility check:
y(1) 4.000000
y(n-1) 4.000000
```

You can use these CPU examples and port them to the GPU on your own to get more experience at a later point in time. We will step 
through the process in these exercises to show you how it is done.

First we will go back the C example directory and work with a very simple case. It has all the code in a single subroutine with 
statically allocated arrays on the stack. This permits the compiler to have as much information as possible. Note that we could 
also reload the regular amdclang module instead of the new amdflang beta. But the amdflang also has a perfectly good amdclang
compiler. Also, we have made the array size smaller so that it won't run out of stack space.

```
cd ../../C/SingleLineConstructs
make saxpy_gpu_singleunit_static
./saxpy_gpu_singleunit_static
```

You will get a warning about vectorization that is telling you that you do not need the simd clause for the amdclang compiler. But
it compiles fine and creates an executable. We run the executable.

```
./saxpy_gpu_singleunit_static
```

The output

```
Time of kernel: 0.016511
check output:
y[0] 4.000000
y[N-1] 4.000000
```

We note that we did not have to supply any explicit memory management such as a map clause. The compiler can detect the array sizes
and that the arrays need to be moved.

Now let's move on to the next example where we dynamically allocate the arrays. We are still using a single subroutine as the previous
example.

```
make saxpy_gpu_singleunit_dynamic
./saxpy_gpu_singleunit_dynamic
```

This time we get the follow output on a MI200 series GPU. 

```
Queue error - HSA_STATUS_ERROR_MEMORY_FAULT
Display only launched kernel:
Kernel 'omp target in main @ 19 (__omp_offloading_34_4474430_main_l19)'
OFFLOAD ERROR: Memory access fault by GPU 8 (agent 0x5ebda70) at virtual address 0x7f81e79dd000. Reasons: Unknown (0)
Use 'OFFLOAD_TRACK_ALLOCATION_TRACES=true' to track device allocations
Aborted (core dumped)
```

The error message makes it very clear that we are missing the data for the array. We could follow the advice to get a
more detailed report if we do not know what array it is. But we'll take a simpler approach. We'll set the 
`HSA_XNACK` environment variable to tell the system to manage the memory for us. This will work on the data center 
AMD Instinct GPUs. For workstation GPUs, you may need to add an explicit map clause.

```
export HSA_XNACK=1
./saxpy_gpu_singleunit_dynamic
```

Now we get the expected output:

```
Time of kernel: 0.063025
check output:
y[0] 4.000000
y[N-1] 4.000000
```

So the compiler can sometimes help with moving the memory in very simple cases. But it doesn't take much complexity before
it doesn't have enough information. We return to our original `saxpy_cpu.c` example and change the pragma to direct the
compiler to offload the calculation to the GPU as already done in `saxpy_gpu_parallelfor.c. We keep the `HSA_XNACK=1`
setting from before.

```
#pragma omp target teams distribute parallel for simd
```

And building and running the example.

```
make saxpy_gpu_parallelfor
./saxpy_gpu_parallelfor
```

Output

```
Time of kernel: 0.061191
check output:
y[0] 4.000000
y[N-1] 4.000000
```

OpenMP has added a simpler loop directive that you can also use. The pragma line is pretty long for
the original directive, so this should make it simpler to add to your program. The new pragma is

```
#pragma omp target teams loop
```

This form generally will produce the same results as the earlier directive. But, in principle, it
may give the compiler more freedom how to generate the parallel GPU code.

```
make saxpy_gpu_loop
./saxpy_gpu_loop
```

Even the example is a bit easier to run with less typing.

The output 

```
Time of kernel: 0.061429
check output:
y[0] 4.000000
y[N-1] 4.000000
```

So now we have demonstrated how easy it is to add a pragma to a loop to cause it to run on the GPU. And we have seen a
little on how the managed memory capability makes the process a little easier. We can focus on parallelizing each
loop rather than worrying about where our array data is located. 

You can experiment with these examples on both a MI300A APU and a discrete GPU such as MI300X or MI200 series GPU. You
should see a performance difference since the MI300A only has to map the pointer and not move the whole array.

We have one less example to look at. Many scientific codes have multi-dimensional data that need to be operated on.
We can use the collapse clause to spread out the work from both loops rather than just the outer one. This can 
be helpful if the outer loop is small. But since we are always trying to generate more work and parallelism, it
can also have some benefit for larger outer loops.

We'll go back to our Fortran example directory since 2-dimensional arrays are much easier to work with in Fortran.
The directive will now become

```
!$omp target teams distribute parallel do collapse(2)
```

Building and running the example

```
cd ../../Fortran/SingleLineConstructs
make saxpy_gpu_collapse
./saxpy_gpu_collapse
```

And the output

```
Time of kernel: 0.029263
 plausibility check:
y(1,1) 4.000000
y(m,n) 4.000000
```

There are now Fortran equivalents for most of the same cases. You can try them as well. All of them will work without
HSA_XNACK being set. The reason is that Fortran passes the array size information along with the array. So the compiler
has more information to work with. In Fortran, the additional information is called the "dope" vector. It is last
century slang for "give me the dope on it". We would say "beta" in today's slang. 