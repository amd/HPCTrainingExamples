Environment variables:
  `CXX=<C++ Compiler>`
      Example: `CXX=amdclang++`
  `LIBOMPTARGET_INFO=-1`
  `LIBOMPTARGET_KERNEL_TRACE=[1,2]`
  `OMP_TARGET_OFFLOAD=MANDATORY`

Mem1 pattern : Typical example with single map clause at computational loop

   Map clause on pragma line just before computational loop
   `mem1.cc:#pragma omp target teams distribute parallel for simd map(to: x[0:n], y[0:n]) map(from: z[0:n])`

Mem2 pattern : Add enter/exit data alloc/delete when memory is created/freed

   After new
   `mem2.cc:#pragma omp target enter data map(alloc: x[0:n], y[0:n], z[0:n])`

   Loop around computational loop and keep map on computational loop. The map to/from should check if the
   data exists. If not, it will allocate/delete it. Then it will do the copies to and from. This will
   increment the Reference Counter and decrement it at end of loop.
   `mem2.cc:#pragma omp target teams distribute parallel for simd map(to: x[0:n], y[0:n]) map(from: z[0:n])`

   Before delete
   `mem2.cc:#pragma omp target exit data map(delete: x[0:n], y[0:n], z[0:n])`

Mem3 pattern: Replacing map to/from with updates to bypass unneeded device memory check

   After new
   `mem3.cc:#pragma omp target enter data map(alloc: x[0:n], y[0:n], z[0:n])`

   Before computational loop. Data should be copied. Reference counter should not change.
   `mem3.cc:#pragma omp target update to (x[0:n], y[0:n])`
   `mem3.cc:#pragma omp target teams distribute parallel for simd`

   After computational loop
   `mem3.cc:#pragma omp target update from (z[0:n])`

   Before delete
   `mem3.cc:#pragma omp target exit data map(delete: x[0:n], y[0:n], z[0:n])`

Mem4 pattern: Replacing delete with release to use Reference Counting

```
   mem4.cc:#pragma omp target enter data map(alloc: x[0:n], y[0:n], z[0:n])
   mem4.cc:#pragma omp target exit data map(release: x[0:n], y[0:n], z[0:n])
   mem4.cc:#pragma omp target teams distribute parallel for simd map(to: x[0:n], y[0:n]) map(from: z[0:n])
```

Mem5 pattern: Using enter data map to/from alloc/delete to reduce memory copies

```
   mem5.cc:#pragma omp target enter data map(to: x[0:n], y[0:n]) map(alloc: z[0:n])
   mem5.cc:#pragma omp target exit data map(from: z[0:n]) map(delete: x[0:n], y[0:n])
   mem5.cc:#pragma omp target teams distribute parallel for simd map(to:x[0:n], y[0:n]) map(from: z[0:n])
```

Mem6 pattern: Using enter data alloc/delete with update clause at end

```
   mem6.cc:#pragma omp target enter data map(alloc: x[0:n], y[0:n], z[0:n])
   mem6.cc:#pragma omp target teams distribute parallel for simd
   mem6.cc:#pragma omp target update from(z[0])
   mem6.cc:#pragma omp target exit data map(delete: x[0:n], y[0:n], z[0:n])
   mem6.cc:#pragma omp target teams distribute parallel for simd
```

Mem7 pattern: Using Unified Shared Memory to automatically move data

```
   set HSA_XNACK=1 at runtime
   mem7.cc:#pragma omp requires unified_shared_memory
   mem7.cc:#pragma omp target teams distribute parallel for simd
   mem7.cc:#pragma omp target teams distribute parallel for simd
```

Mem8 pattern: Demonstrating Unified Shared Memory with maps for backward compatibility

```
   set HSA_XNACK=1 at runtime
   mem8.cc:#pragma omp requires unified_shared_memory
   mem8.cc:#pragma omp target enter data map(alloc: x[0:n], y[0:n], z[0:n])
   mem8.cc:#pragma omp target teams distribute parallel for simd
   mem8.cc:#pragma omp target update from(z[0])
   mem8.cc:#pragma omp target exit data map(delete: x[0:n], y[0:n], z[0:n])
   mem8.cc:#pragma omp target teams distribute parallel for simd
```

Mem9 pattern: Using std::vector with Unified Shared Memory to automatically move data

```
   mem9.cc:#pragma omp requires unified_shared_memory
   mem9.cc:#pragma omp target teams distribute parallel for simd
   mem9.cc:#pragma omp target teams distribute parallel for simd
```

Mem10 pattern: Demonstrating Unified Shared Memory with valarray and maps for backward compatibility

```
   mem10.cc:#pragma omp requires unified_shared_memory
   mem10.cc:#pragma omp target enter data map(alloc: xptr[0:n], yptr[0:n], zptr[0:n])
   mem10.cc:#pragma omp target teams distribute parallel for simd
   mem10.cc:#pragma omp target update from(zptr[0])
   mem10.cc:#pragma omp target exit data map(delete: xptr[0:n], yptr[0:n], zptr[0:n])
   mem10.cc:#pragma omp target teams distribute parallel for simd
```

Mem11.cc Adding memory alignment to Mem 8 code with unified shared memory and backwards compatibility

```
   mem11.cc:#pragma omp requires unified_shared_memory
   mem11.cc:#pragma omp target enter data map(alloc: x[0:n], y[0:n], z[0:n])
   mem11.cc:#pragma omp target teams distribute parallel for simd
   mem11.cc:#pragma omp target update from(z[0])
   mem11.cc:#pragma omp target exit data map(delete: x[0:n], y[0:n], z[0:n])
   mem11.cc:#pragma omp target teams distribute parallel for simd
```
