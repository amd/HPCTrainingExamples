
## Porting exercise reduction of multiple scalars in one kernel

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/4_reduction_scalars` from the Training Examples repository.

This folder has two code versions:

### 0) A serial cpu version to port yourself.

Hint: don't forget to port the Makefile.

Build:
```
make
````
Run:
```
./reduction_scalar
```

### 1) An openmp offload ported solution.
It shows how you can do a reduction of multiple scalars in one kernel. Note that scalars do not need to be explicitly mapped.
