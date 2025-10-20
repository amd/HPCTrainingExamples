
## Porting a small app: jacobi

Note: This exercise code and solutions were tested with amdflang-new. Using cray ftn is also possible. Setup the environment accordingly.

The C++ version of this code is described in detail https://rocm.blogs.amd.com/high-performance-computing/jacobi/README.html

The Fortran code is described here: https://rocm.blogs.amd.com/ecosystems-and-partners/fortran-journey/README.html
```
cd 0_jacobi_portyourself
```
decide if you want to port with or without unified shared memory. <hr>

**Either** for USM
```
export HSA_XNACK=1
```
<hr>

**or** without for a behaviour with copies
```
export HSA_XNACK=0
```
<hr>

```
make
```
run the code for a reference CPU solution.
```
./jacobi
```
you can specify a problem size with
```
./jacobi -m <Problemsize>
```
otherwise a default of ```<Problemsize> = 4096``` is set.
On the CPU the default problem size may take a minute to get solved, on the GPU it should be solved in a matter of second(s).

- port the Makefile
- require unified memory (in case ```export HSA_XNACK=1``` is set)
- port the loops
- test functionality and output after each loop you ported
- make sure the kernels really run on the GPU
- introduce map clauses (in case ```export HSA_XNACK=0``` is set)
- optimize data movement (in case ```export HSA_XNACK=0``` is set)

### Example solutions:
For the unified shared memory version:
```
export HSA_XNACK=1
cd 1_jacobi_usm
make
./jacobi
```
compare with your solution (e.g. use ```vimdiff``` to compare).

For the discrete GPU / manual memory management version:
```
export HSA_XNACK=0
cd 2_jacobi_targetdata
make
./jacobi
```
compare with your solution (e.g. use ```vimdiff``` to compare).
