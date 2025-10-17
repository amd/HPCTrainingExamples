
## Porting exercise: reduction

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/C/2_reduction` from the Training Examples repository.

This exercise focuses on two things:
- Part 1: how to port a reduction to the GPU
- Part 2: importance of map clauses on discrete GPUs or ```HSA_XNACK=0``` on MI300A
  
First, prepare the environment (loading modules, set environment variables), if you didn't do so before.

### For Part 1 and 2: serial CPU version to port
The CPU version to port yourself is located in:
```
cd 0_reduction_portyourself
vi reduction.c
```
Only port the Makefile and the reduction itself! This exercise focuses on how to implement a reduction, not on porting the full example.

How to build all versions:
```
make
```
and run:
```
./reduction
```
The other folders 1 and 2 have different flavors of the solution:

### Part 1: Port with unified shared memory
```
cd 1_reduction_solution_usm
vi reduction.c
```
contains a sample solution for unified shared memory / the APU programming model and requries ```export HSA_XNACK=1``` to be set in advance.
The correct output is `200000` for each element.

### Part 2: Port with map clause
#### 2.1 Porting exercise
```
cd 2_reduction_solution
vi reduction.c
```
contains a sample solution for discrete GPUs and requries ```export HSA_XNACK=0``` to be set in advance.
The correct output is `200000` for each element.

#### 2.2 Behaviour with and without USM
The third folder contains an exercise to explore the behavior with and without USM:
```
cd 3_reduction_wrongmapping
vi reduction.c
```
This example intentionally does the mapping wrong (```from``` instead of ```to```). You can see how the result changes (output 0 instead of 20000) when you use export ```export XSA_XNACK=0```. No error is shown, but the result is wrong. 
Test the same wrong code with ```export HSA_XNACK=1```, then the result is correct again as mapping clauses are ignored.
Take home message: if you develop for both APUs and discrete GPUs on MI300A, check if the results are the same for ```HSA_XNACK=0``` and ```=1``` as map clauses will be ignored with ```HSA_XNACK=1```! Ignoring memory copies is great for code portability and performance without code changes, but be careful to include proper validation checks during development for both discrete GPUs and APUs.

