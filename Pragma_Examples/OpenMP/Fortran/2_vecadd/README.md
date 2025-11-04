
## Porting exercise: vecadd

This exercise combines what you learned in the exercises 1_saxpy and 3_reduction (recommended to do those first): porting simple kernels without and with reduction.

## Serial CPU code to port
The CPU code to port yourself can be found in
```
cd 0_vecadd_portyourself
```
Don't forget to port the Makefile, too, and validate your application runs on the GPU (see 1_saxpy example)!
Build with
```  
make
```
and run
```
./vecadd
```
Remember the output result for the serial version to validate the offload version.

Port the example (including the Makefile) in the following and build and run the example after every kernel you ported to ensure correctness.

## Part 1: unified shared memory
The following example port assumes unified shared memory.
Set 
```
export HSA_XNACK=1
```
```
cd 1_vecadd_usm  
```
contains the unified shared memory solution. 

Note: if you set ```HSA_XNACK=0``` you will get an error message. This is the intended behaviour for USM, if XNACK is disabled. This is not yet always the case for the beta release of amdflang-new and will be fixed in a future release. To enforce that unified shared memory is properly recognized, or if you don't want to add it in every module by hand, you can use the compiler flag ```-fopenmp-force-usm```. With this the Next Generation Fortran Compiler beta release behaves as intended by the OpenMP standard.

## Part 2: with map clauses
Port first with map clauses and later optimize the data movement:
Set 
```
export HSA_XNACK=0
```
A solution with simple map clauses is contained here:
```
cd 2_vecadd_map
```
and this solution moves data only where necessary:
```
cd 3_vecadd_targetdata
```

## Part 3: asynchronous offload
This secton presents a solution for asynchronous offload both with USM
```
cd 4_vecadd_usm_async
```
and without USM
```
cd 5_vecadd_async
```

In the solution, the first kernel was split to show the functionality better. Investigate what happens, if you would remove the ```!$omp taskwait``` directive.
If you remove the ```!$omp taskwait``` the result will be wrong as the CPU does not wait for the GPU to synchronize before the data is accessed on the CPU.
