
## Excercise: mapping of different datatypes

README.md in `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/7_derived_types` of the Training Exercises repository.

This excercise explores the possibilities of mapping derived types. This is one of the main challenges one may encounter when porting a Fortran app to discrete GPUs. This excercise also shows that on the APU using `HSA_XNACK=1` such problems do not exist.
Note: This excercise was designed for amdflang-new.

Compile the examples:
```
make
```
first, set 
```
export HSA_XNACK=0
```
to explore the behaviour similar to a discrete GPU (Remark: ```vimdiff file1 file2``` may help to find the differences).

Explore and run the four examples:

1) The first example leaves the mapping to the compiler

Run:
```
./dtype_derived_type_automap
```
this results in a memory access fault. Hence, this implementation is wrong on a discrete GPU (or MI300A: disabling HSA_XNACK).

3) The second example adds mapping clauses for the allocatable array which is a member of the derived type

Run:
```
./dtype_derived_type
```
this again results in a memory access fault

5) The third example provides a solution: a pointer to the allocatable array is introduced

Run:
```
./dtype_pointer
```

6) In example 2 and 3 the scalars used for the range of the loop were replaced by integer numbers to see the impact of the allocatable array only. In this forth example they are re-introduced. This example shows, that mapping of scalar members of derived types is working.

Run:
```
./dtype_scalar_members
```

8) When you run the unified shared memory version with XNACK off, you will get a warning and the same memory access fault as in example 1 and two
```
./dtype_derived_type_usm
```
AMDGPU message: Running a program that requires XNACK on a system where XNACK is disabled. This may cause problems when using an OS-allocated pointer inside a target region. Re-run with HSA_XNACK=1 to remove this warning.

Now switch on unified shared memory by 
```
export HSA_XNACK=1
```
Run all the five examples again. All of them should run sucessfully.

Set 
```
export LIBOMPTARGET_INFO=-1 
```
with the amdflang-new compiler or 
```
export CRAY_ACC_DEBUG=1

```
if you work with the ftn compiler.

Run example 3  with and without unified shared memory (export HSA_XNACK=1 and  HSA__XNACK=0)
You are able to see host to device copies in the shown log in the case of HSA_XNACK=0.
In the case of HSA_XNACK=1 those copies are gone and this message is shown:

AMDGPU device 0 info: Application configured to run in zero-copy using auto zero-copy.

Hence, if a discrete GPU program is compiled with HSA_XNACK=1 on MI300A, memory copies are automatically ignored. This makes code portable between discrete GPUs an APUs. Include !$omp requires_unified_shared_memory at the top of the program (after implicit none) such that the compiler can make full use of the APU programming model. This is shown in example code 5.
When you compare the code examples, the unified_shared_memory version dtype_derived_type_usm (version 5) is very simple to implement. If you only work on an APU, this is the easiest way to port, as mapping clauses are not required to obtain good performance.

You may want to set
```
export LIBOMPTARGET_INFO=0
```
before you run the next excercise.
