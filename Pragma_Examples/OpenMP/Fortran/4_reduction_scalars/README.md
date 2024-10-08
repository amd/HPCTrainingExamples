# Porting excercise reduction of multiple scalars in one kernel

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

1) an openmp offload ported solution. It shows how you can do a reduction of multiple scalars in one kernel. Note that scalars do not need to be explicitly mapped.
