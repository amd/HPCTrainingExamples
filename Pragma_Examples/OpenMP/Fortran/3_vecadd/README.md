# Porting excercise: vecadd

This excercise combines what you learned in the previous two excercises: porting simple kernels without and with reduction.
# For Part 1 and Part 2: Serial CPU code to port
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
Set ```export HSA_XNACK=1```.
```
cd 1_vecadd_usm  
```
contains the unified memory solution. Note: if you set ```HSA_XNACK=0``` you will get an error message. This is the intended behaviour for USM, if XNACK is disabled. This is not yet always the case for the beta release of amdflang-new.
# Part 2: with map clauses
Set ```export HSA_XNACK=0```.
```
cd 2_vecadd_map
```
contains a solution with simple map clauses.
```
cd 3_vecadd_targetdata
```
This solution moves data only were necessary.
