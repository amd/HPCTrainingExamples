
## OpenMP Single Line Compute Constructs:

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/SingleLineConstructs` in the Training Examples repository

We start with adding a single line directive to move the computation of a loop to the GPU. The exercises for this will utilize
the saxpy example.

`NOTE`: the examples in Fortran also work without setting `HSA_XNACK=1`. The reason is that Fortran passes the array size information along with the array. So the compiler has more information to work with. In Fortran, the additional information is called the "dope" vector. It is last
century slang for "give me the dope on it". We would say "beta" in today's slang.


### CPU version

This example uses OpenMP on the CPU with threading for parallelism. The pragma used is

```
#pragma omp parallel for
```

We go to the directory with the example and load the amdclang module. We can then build and run the code.

```
cd HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/SingleLineConstructs
module load amdflang-new
make saxpy_cpu
./saxpy_cpu
```

You should get some output like:

```
Time of kernel: 0.151135
plausibility check:
y(1) 4.000000
y(n-1) 4.000000
```

You can use these CPU examples and port them to the GPU on your own to get more experience at a later point in time. We will step
through the process in these exercises to show you how it is done.

First we will work with a very simple case. It has all the code in a single subroutine with
statically allocated arrays on the stack. This permits the compiler to have as much information as possible. Note that we could
also load the regular amdclang module instead of the new amdflang.
Also, we have made the array size smaller so that it won't run out of stack space.

```
make saxpy_gpu_singleunit_autoalloc
./saxpy_gpu_singleunit_autoallloc
```

The output

```
Time of kernel: 0.022465
 plausibility check:
y(1) 4.000000
y(n) 4.000000
```

We note that we did not have to supply any explicit memory management such as a map clause. The compiler can detect the array sizes
and that the arrays need to be moved.

Now let's move on to the next example where we dynamically allocate the arrays. We are still using a single subroutine as the previous
example. Note that, unlike the C case, we are not setting `HSA_XNACK=1` to make the example run (see note at the beginning of this README):

```
make saxpy_gpu_singleunit_dynamic
./saxpy_gpu_singleunit_dynamic
```

This time we get the following output:

```
Time of kernel: 0.022440
 plausibility check:
y(1) 4.000000
y(n) 4.000000
```

We return to our original `saxpy_cpu.c` example and change the pragma to direct the
compiler to offload the calculation to the GPU as already done in `saxpy_gpu_paralleldo.F90`.
setting from before.

```
#pragma omp target teams distribute parallel for simd
```

And building and running the example.

```
make saxpy_gpu_paralleldo
./saxpy_gpu_paralleldo
```

Output

```
Time of kernel: 0.052156
 plausibility check:
y(1) 4.000000
y(n) 4.000000
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
Time of kernel: 0.052010
 plausibility check:
y(1) 4.000000
y(n) 4.000000
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

We'll consider the case of Fortran since 2-dimensional arrays are much easier to work with.
The directive will now become

```
!$omp target teams distribute parallel do collapse(2)
```

Building and running the example

```
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
