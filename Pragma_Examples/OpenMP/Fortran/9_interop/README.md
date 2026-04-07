# Running a Fortran to HIP interop example

README: `Pragma_Examples/OpenMP/Fortran/9_interop`

This is a simple example to demonstrate Fortran to HIP interoperability.
Load a current ROCm version (at least rocm/7.0) and build with
```
module load rocm/7.2.0      # AAC6
module load rocm-new/7.2.0  # AAC7
make
```
Make sure to choose the correct module for the system you are on and run it with
```
./interop
```
The code will run to completion if it passes verification
