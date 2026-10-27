
## Porting exercise reduction into an array

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/5_reduction_array` from the Training Examples repository.

This folder has three code versions:

### 0) Serial cpu version to port yourself.

Hint: don't forget to port the Makefile.

Build:
```
make
````
Run:
```
./reduction_array
```

### 1) An openmp offload ported solution.
The solution shows how you can do a reduction of multiple values into an array in one kernel.

### 2) Additional min/max/sum reduction tests
The folder `2_reduction_array_min_max_sum` contains self-checking tests (each prints `PASS` or `FAIL`) for `min`, `max` and `sum` reductions on integer and real arrays. It builds with `make` and runs with `./reduction`.
