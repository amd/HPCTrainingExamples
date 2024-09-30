# Porting excercise: vecadd

This excercise combines what you learned in the previous two excercises: porting simple kernels without and with reduction.
Decide, if you want to use the APU or discrete GPU programming model. Set export ```HSA_XNACK=1``` or ```=0``` appropriately.

0) CPU code to port yourself. Don't forget to port the Makefile, too, and validate you application runs on the GPU!
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
remember the output result for the serial version to validate the offload version.
adapt Makefile for offload
port the example, build and run after every kernel you ported to ensure correctness.

1-3) Solution
The other three folders contain different flavors of the solution. Build and run instructions are the same as for the serial version. The Makefiles are already ported.
1_vecadd_usm  - unified memory version don't forget to set ```HSA_XNACK=1```
2_vecadd_map  - simple mapping clauses, set ```HSA_XNACK=0``` to observe behaviour similar to discrete GPUs
3_vecadd_targetdata - move data only were necessary, set ```HSA_XNACK=0``` to observe behaviour similar to discrete GPUs

