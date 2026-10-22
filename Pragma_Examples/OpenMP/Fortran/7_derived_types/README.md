
## Exercise: mapping of different datatypes

README.md in `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/7_derived_types` of the Training Exercises repository.

This exercise explores the possibilities of mapping derived types. This is one of the main challenges one may encounter when porting a Fortran app to discrete GPUs. This exercise also shows that on the APU using `HSA_XNACK=1` such problems do not exist.
Note: This exercise targets the new LLVM-based **`amdflang`** compiler, which is shipped starting with ROCm/7.0.

Compile the examples:
```
make
```
first, set
```
export HSA_XNACK=0
```
to explore the behaviour similar to a discrete GPU (Remark: ```vimdiff file1 file2``` may help to find the differences).

Explore and run the five examples:

### 1) The first example leaves the mapping to the compiler

Run:
```
./dtype_derived_type_automap
```
this results in a memory access fault. Hence, this implementation is wrong on a discrete GPU (or MI300A: disabling HSA_XNACK).

### 2) The second example adds mapping clauses for the allocatable array which is a member of the derived type

Run:
```
./dtype_derived_type
```
this again results in a memory access fault

### 3) The third example provides a solution: a pointer to the allocatable array is introduced

Run:
```
./dtype_pointer
```

### 4) In example 2 and 3 the scalars used for the range of the loop were replaced by integer numbers to see the impact of the allocatable array only. In this fourth example they are re-introduced. This example shows, that mapping of scalar members of derived types is working.

Run:
```
./dtype_scalar_members
```

### 5) When you run the unified shared memory version with XNACK off, you will get a warning and the same memory access fault as in examples 1 and 2
```
./dtype_derived_type_usm
```
AMDGPU message: Running a program that requires XNACK on a system where XNACK is disabled. This may cause problems when using an OS-allocated pointer inside a target region. Re-run with HSA_XNACK=1 to remove this warning.

Now switch on unified shared memory by 
```
export HSA_XNACK=1
```
Run all the five examples again. All of them should run successfully.

For `amdflang` set
```
export LIBOMPTARGET_INFO=-1 
```

Run example 3 with and without unified shared memory (`export HSA_XNACK=1` and `export HSA_XNACK=0`)
You are able to see host to device copies in the shown log in the case of HSA_XNACK=0.
In the case of `HSA_XNACK=1` those copies are gone and this message is shown:

```
AMDGPU device 0 info: Application configured to run in zero-copy using auto zero-copy.
```

Hence, if a discrete GPU program is compiled with `HSA_XNACK=1` on MI300A, memory copies are automatically ignored. This makes code portable between discrete GPUs and APUs. Include `!$omp requires unified_shared_memory` at the top of the program (after `implicit none`) such that the compiler can make full use of the APU programming model. This is shown in example code 5.

