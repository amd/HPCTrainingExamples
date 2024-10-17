# Porting excercise: vecadd

This excercise combines what you learned in the previous two excercises: porting simple kernels without and with reduction.
# Serial CPU code to port
CPU code to port yourself. Don't forget to port the Makefile, too, and validate you application runs on the GPU!
```
cd 0_vecadd_portyourself
```
Build
```  
make
```
run
```
./vecadd
```
Remember the output result for the serial version to validate the offload version.
adapt Makefile for offload
port the example, build and run after every kernel you ported to ensure correctness.

# Part 1: unified shared memory
Port assuming unified shared memory.
Set ```export HSA_XNACK=1```.
```
cd 1_vecadd_usm  
```
contains the unified memory solution. Note: if you set ```HSA_XNACK=0``` you will get an error message. This is the intended behaviour for USM, if XNACK is disabled. This is not yet always the case for the beta release of amdflang-new.

# Part 2: with map clauses
Port first with map clauses and later optimize the data movement:
Set ```export HSA_XNACK=0```.
```
cd 2_vecadd_map
```
contains a solution with simple map clauses.
```
cd 3_vecadd_targetdata
```
This solution moves data only were necessary.

# Part 3: asynchronous offload
A solution for asynchronous offload with USM. 

In the solution the first kernel was split to show the functionality better. Investigate what happens, if you would remove the ```!$omp taskwait```.
```
cd 4_vecadd_usm_async
```
and without
```
cd 4_vecadd_async
```
If you remove the ```!$omp taskwait``` the result will be wrong as the CPU does not wait for the GPU to synchronize before the data is accesses on the CPU.
