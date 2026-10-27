
## Submodule test -- does the Fortran compiler support the new submodules feature in the Fortran 2008 standard (extension in 2003)

README.md in `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/Submodules` of the Training Exercises repository.

This exercise checks whether the loaded Fortran compiler supports submodules.
With the ROCm 10.1 `amdflang` compiler, run:

```
module load rocm/10.1.0
export FC=amdflang
make
```

The build should compile both `interface.f90` and `impl.f90` successfully.
