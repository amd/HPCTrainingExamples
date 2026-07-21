
## Submodule test -- does the Fortran compiler support the new submodules feature in the Fortran 2008 standard (extension in 2003)

README.md in `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/Submodules` of the Training Exercises repository.

Try with the old flang compiler -- Load the amdclang module and you get

```
make
```

```
/opt/rocm-6.4.0/llvm/bin/amdflang -c -g -O3 -fopenmp interface.f90
/opt/rocm-6.4.0/llvm/bin/amdflang -c -g -O3 -fopenmp impl.f90
F90-S-1059-The definition of subprogram module_func_impl does not have the same number of arguments as its declaration (impl.f90: 6)
  0 inform,   0 warnings,   1 severes, 0 fatal for module_func_impl
make: *** [Makefile:11: impl.o] Error 1
```

Try with a recent next generation flang compiler

```
make
```

```
/opt/rocmplus-6.4.0/rocm-afar-6.1.0/bin/amdflang -c -g -O3 -fopenmp interface.f90
/opt/rocmplus-6.4.0/rocm-afar-6.1.0/bin/amdflang -c -g -O3 -fopenmp impl.f90
```
