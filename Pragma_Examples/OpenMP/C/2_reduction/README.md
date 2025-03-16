# Reduction excercise:

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/C/2_reduction` from the Training Examples repository.

This excercise will show how to port a reduction.

#### 0) serial CPU version 
Version to port yourself. Don't forget to port the Makefile.
```
cd 0_reduction_portyourself
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
Adapt Makefile for offload.
Port the example, build and run after every kernel you ported to ensure correctness.

#### 1) solution with unified shared memory
Set 
```export HSA_XNACK=1```
to test this version.
```
cd 1_reduction_usm
```
Build
```  
make
```
run
```
./reduction
```
Note: you may want to use ```vimdiff <file1> <file2>``` to compare your solution with this version.

#### 2) solution with map clauses
Set 
```export HSA_XNACK=0```
to test this version.
```
cd 2_reduction_map
```
Build
```  
make
```
run
```
./reduction
```
Note: you may want to use ```vimdiff <file1> <file2>``` to compare your solution with this version.


