# Porting excercise: vecadd

This excercise combines what you learned in the previous two excercises: porting simple kernels without and with reduction.
Decide, if you want to use the APU or discrete GPU programming model. Set export ```HSA_XNACK=1``` or ```=0``` appropriately.

#### 0) serial CPU version
version to port yourself. Don't forget to port the Makefile, too, and validate you application runs on the GPU!
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

1-3) Solution
The other three folders contain different flavors of the solution. Build and run instructions are the same as for the serial version. The Makefiles are already ported.

#### 1) Version with unified shared memory
```
cd 1_vecadd_usm
````
Build and run like above.
Unified memory version don't forget to set ```export HSA_XNACK=1```. 

Note: if you set ```HSA_XNACK=0``` you will get an error message. This is the intended behaviour for USM, if XNACK is disabled. This is not yet always the case for the beta release of amdflang-new.

#### 2) with map clauses
```
2_vecadd_map
```
Build and run like above.
Version with simple mapping clauses, set ```export HSA_XNACK=0``` to observe behaviour similar to discrete GPUs

#### 3) with data region
```
3_vecadd_targetdata
```
Build and run like above.
Move data only were necessary a structured data region is used here, set ```export HSA_XNACK=0``` to observe behaviour similar to discrete GPUs

#### 4) with unified shared memory and asynchronous execution
```
4_vecadd_async_usm
```
Build and run like above.
Unified memory version with asynchronous OpenMP kernel calls. Don't forget to set ```export HSA_XNACK=1```.

#### 5) with map clauses and asynchronous execution
```
5_vecadd_async
```
Build and run like above.
Version with a `target enter data` map clause directive and asynchronous OpenMP kernel calls. Set ```export HSA_XNACK=0``` to observe behaviour similar to discrete GPUs.
