
## Porting exercise reduction of multiple scalars in one kernel

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/C/4_reduction_scalars` from the Training Examples repository.

This folder has two code versions:

#### 0) a serial cpu version 
to port yourself. 
Hint: don't forget to port the Makefile.

Build:
```
make
````
Run:
```
./reduction_scalar
```

#### 1) an openmp offload ported solution. 
The solution shows how you can do a reduction of multiple scalars in one kernel. Note that scalars do not need to be explicitly mapped.
