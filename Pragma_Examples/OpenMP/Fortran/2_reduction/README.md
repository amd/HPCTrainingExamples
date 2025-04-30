# Porting exercise: reduction

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/2_reduction` from the Training Examples repository.

This excercise focusses on two things:
- Part 1: how to port a reduction to the GPU with the APU programming model
- Part 2: importance of map clauses on discrete GPUs or when using ```HSA_XNACK=0``` on MI300A
  
First, prepare the environment (loading modules, set environment variables), if you didn't do so before.
## For Part 1 and 2: serial CPU version to port
0) a version to port yourself.
```
cd 0_reduction_portyourself
vi freduce.F
```
- Only port the Makefile and the reduction itself. This excercise focusses on how to implement a reduction, not on porting the full example.

How to build all versions:
```
make
```
and run:
```
./freduce
```
The other folders 1 and 2 have different flavors of the solution:
## Part 1: Port with unified shared memory
```
cd 1_reduction_solution_usm
vi freduce.F
```
contains a sample solution for unified shared memory / APU programming model (correct output: each element 1010)  run this with setting ```export HSA_XNACK=1``` in advance

## Part 2: Port with map clause
### 2.1 Porting excercise
```
cd 2_reduction_solution
vi freduce.F
```
Contains a sample solution for discrete GPUs (correct output: each element 1010) run this with setting ```export HSA_XNACK=0``` in advance
### 2.2 Behaviour with and without USM
The third folder contains an excercise to explore the behavior with and without USM:
```
cd 3_reduction_solution
vi freduce.F
```
This example intentionally does the mapping wrong (from instead of to). You can see how the result changes (output 1000 instead of 1010) when you use export ```export XSA_XNACK=0```. No error is shown, but the result is wrong. 
Test the same wrong code with ```export HSA_XNACK=1```, then the result is correct again as mapping clauses are ignored.
Take home message: if you develop for both APUs and discrete GPUs on MI300A, check if the results are the same for ```HSA_XNACK=0``` and ```=1``` as map clauses will be ignored with ```HSA_XNACK=1```! Ignoring memory copies is great for code portability and performance without code changes, but be careful to include proper validation checks during development for both discrete GPUs and APUs.
