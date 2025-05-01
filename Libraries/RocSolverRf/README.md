# Purpose of this code
If you are dealing with a sequence of sparse linear systems (that share THE SAME NON-ZERO PATTERN)

$$
A^{(i)}x^{(i)} = b^{(i)} \quad i = 0, 1, 2, \ldots, 
$$ 

where $A^{(i)}$ is an $N \times N$ matrix, $b^{(i)}$ is an $N\times 1$ right-hand side and $x^{(i)}$ are the vectors, these systems can be solved faster using refactorization. 

Idea: the first system in the sequence is solved on the CPU using a direct solver of your choice (there are two examples in the code, one using KLU and the other one using UMFPACK) and the refactorization is set on the GPU using RocSolverRf library. The following systems are solved using refactorization on the GPU, which is often much faster.

The examples show you how to use rocThrust for data allocation, how to use rocSparse for format conversion and how to set up and use rocSolverRf. 

To run, pick two or three linear systems from a series and add them to the command line (see below). The executable reads matrices in MatrixMarket format. Right hand sides are optional.   

# Compile and Run

To compile and run an example of this code do:

```
../../tests/run_RocSolverRf.sh
``` 

The above script will install Lapack and SuiteSparse in a directory called `dependencies`. It will also download a matrix for a trial run. 

This directory contains both KLU and UMFPACK examples. To run, pick two (or three) matrices from a sequence, with THE SAME sizes and THE SAME sparsity patterns. You can supply right-hand sides too, but this is optional.
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


## Notes About Running on Frontier

It may be necessary to do:

```
module unload craype
module load cmake gcc
```

If the build of SuiteSparse fails, (which WILL happen on Frontier), complaining that GMP is  not available, install GMP and MPFR. Do not worry about the SPEX error.

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




 
