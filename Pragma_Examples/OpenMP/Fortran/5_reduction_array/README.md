
## Porting exercise reduction of multiple scalars in one kernel

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/9_reduction_array` from the Training Examples repository.

This folder has two code versions:

0) a serial cpu version to port yourself. 

Hint: don't forget to port the Makefile.

Build:
```
make
````
Run:
```
./reduction_scalar
```

1) an openmp offload ported solution. 
The solution shows how you can do a reduction of multiple values into an array in one kernel.
