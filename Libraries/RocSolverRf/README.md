# Purpose of this code
If you are dealing with a sequence of sparse linear systems (that share THE SAME NON-ZERO PATTERN)

$$
A^{(i)}x^{(i)} = b^{(i)} \quad i = 0, 1, 2, \ldots, 
$$ 

where $A^{(i)}$ is an $N \times N$ matrix, $b^{(i)}$ is an $N\times 1$ right-hand side and $x^{(i)}$ are the vectors, these systems can be solved faster using refactorization. 

Idea: the first system in the sequence is solved on the CPU using a direct solver of your choice (there are two examples in the code, one using KLU and the other one using UMFPACK) and the refactorization is set on the GPU using RocSolverRf library. The following systems are solved using refactorization on the GPU, which is often much faster.

The examples show you how to use rocThrust for data allocation, how to use rocSparse for format conversion and how to set up and use rocSolverRf. 

To run, pick two or three linear systems from a series and add them to the command line (see below). The executable reads matrices in MatrixMarket format. Right hand sides are optional.   



RocSolverRf allows to use refactorization.

# Compiling and installing

## Installation


If SuiteSparse is not available as a module, install it first (see below) and modify Makefile accordingly to provide location of the `lib` folder and `include` folder.

```
git clone https://github.com/kswirydo/rocsolverrf_example.git
module load rocm openblas suite-sparse
make
```

If the compilation fails because `umfpack.h` or `klu.h` is not found, or corresponding libraries are missing, than SuiteSparse might not be installed or linked correctly.

Troubleshooting:

1. If using suitesparse module, type `echo $LD_LIBRARY_PATH`. If you see SuiteSparse on the path, copy the path to the library (without the `lib` or `lib64`) and paste in Makefile to define `SS_MAIN` (SuiteSparse main) and recompile.

2. If installed from sources, modify the Makefile and provide the correct `SS_MAIN` path.

## Running

The repository contains both KLU and UMFPACK examples.


To run, pick two (or three) matrices from a sequence, with THE SAME sizes and THE SAME sparsity patterns. You can suppy right-hand sides too, but this is optional.
```
 ./umfpack_example --matrix1 /path/to/matrix1.mtx --matrix2 /path/to/matrix2.mtx 
 ./umfpack_example --matrix1 /path/to/matrix1.mtx --matrix2 /path/to/matrix2.mtx --matrix3 /path/to/matrix3.mtx
 ./klu_example --matrix1 /path/to/matrix1.mtx --matrix2 /path/to/matrix2.mtx 
 ./klu_example --matrix1 /path/to/matrix1.mtx --matrix2 /path/to/matrix2.mtx --matrix3 /path/to/matrix3.mtx
```
or 
```
 ./umfpack_example --matrix1 /path/to/matrix1.mtx --rhs1 /path/to/rhs1.mtx --matrix2 /path/to/matrix2.mtx -rhs2 /path/to/rhs2.mtx
 ./klu_example --matrix1 /path/to/matrix1.mtx --rhs1 /path/to/rhs1.mtx --matrix2 /path/to/matrix2.mtx -rhs2 /path/to/rhs2.mtx
```

Note: if run fails because of missing BLAS, either load a BLAS module (openblas, blas, mkl, lapack, etc) or make sure you have the BLAS you had installed on your path (i.e., that you see the correct path with `echo LD_LIBRARY_PATH`). If not, export to add it.

## Installing SuiteSparse

Skip this step if a) SuiteSparse is already available as a module or b) it was installed by package manager (or from sources).

SuiteSparse requires
- BLAS library (i.e., openBLAS, MKL, LAPACK); often can be pre-loaded as a module.
- GMP (this is a standard library, on many systems and clusters it would already be available).
- MPFR (same as for GMP).

### Installing LAPACK

Make sure cmake and GCC are available. On Frontier:

```
module unload craype
module load gcc cmake
```

Next

```
git clone https://github.com/Reference-LAPACK/lapack
cd lapack
mkdir install
mkdir build 
cd build
cmake -DCMAKE_INSTALL_PREFIX=/path/to/lapack/install -DBUILD_SHARED_LIBS=ON ..
make -j 123
make install
```

In `lapack/install`, you should see `lib64` folder. Add it to the `LD_LIBRARY_PATH`.
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/lapack/install/lib64
```

### Installing SuiteSparse

```
clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git --branch v7.10.1
cd SuiteSparse
mkdir build
mkdir install
```

Open `CMakeList.txt` using vim or other editor and modify lines 38-39 to remove `paru`. I.e., in the original file you have
```
set ( SUITESPARSE_ALL_PROJECTS
    "suitesparse_config;mongoose;amd;btf;camd;ccolamd;colamd;cholmod;cxsparse;ldl;klu;umfpack;paru;rbio;spqr;spex;graphblas;lagraph" )
``` 
and you need to change it to
```
set ( SUITESPARSE_ALL_PROJECTS
    "suitesparse_config;mongoose;amd;btf;camd;ccolamd;colamd;cholmod;cxsparse;ldl;klu;umfpack;rbio;spqr;spex;graphblas;lagraph" )
```
because neither UMFPACK nor KLU need `paru` and it causes compilation errors.

Next

```
cd build
CMAKE_OPTIONS="-DBLA_VENDOR=LAPACK"
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/SuiteSparse/install
```

If this stage fails (which WILL happen on Frontier), compaining that GMP is  not available, install GMP and MPFR first. Do not worry about the SPEX error.


##### Installing GMP
``` 
wget https://gmplib.org/download/gmp/gmp-6.3.0.tar.xz
tar -xf gmp-6.3.0.tar.xz
cd gmp-6.3.0/
./configure --prefix=/path/to/gmp-6.3.0
make -j 123
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/gmp-6.3.0/lib
```

##### Installing MPFR

Note MPFR has GMP as its dependency so they need to be correctly linked together.

```
wget https://www.mpfr.org/mpfr-current/mpfr-4.2.1.tar.xz
tar -xvf mpfr-4.2.1.tar.xz
cd mpfr-4.2.1/
 ./configure --prefix=/path/to/mpfr-4.2.1/ --with-gmp-dir=/path/to/gmp-6.3.0/ --with-gmp-include=/path/to/gmp-6.3.0/include --with-gmp-lib=/path/to/gmp-6.3.0/lib
make -j 123
make install
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/mpfr-4.2.1/lib
```

### Linking SuiteSparse with LAPACK, MPFR and GMP

More than likely, after running `cmake ..` inside `SuiteSparse/build`, it will still report missing GMP/SPEX/MPFR.

To fix, run `ccmake .` and toggle to find `GMP_INCLUDE_DIR`, `GMP_LIBRARY`, and `GMP_STATIC`. Change these entries to `/path/to/gmp-6.3.0/include`, `/path/to/gmp-6.3.0/lib` and `/path/to/gmp-6.3.0/lib`, respectively. Hit `c` to reconfigure. 

If the command fails, reporting `incomplete configuration` and missing MPFR, go back and find `MPFR_INCLUDE_DIR`, `MPFR_LIBRARY` and `MPFR_STATIC`. If you can't find these entries, press `t` to toggle to advanced mode. Change them to `/path/to/mpfr-4.2.1/include`, `/path/to/mpfr-4.2.1/lib` and `/path/to/mpfr-4.2.1/lib`, respectively. Press `c` to reconfigure (you might need to do it twice). Press `g` to `generate`. Exit to the shell and run the following commands. 

```
make -j 123
make install
```

Now both `libklu` and `libumfpack` should be visible in `SuiteSparse/install`. Add this to your `LD_LIBRARY_PATH`
```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/SuiteSparse/install/lib64/
```




 
