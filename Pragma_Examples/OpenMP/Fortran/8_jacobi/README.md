
## Porting a small app: jacobi

Note: This exercise code and solutions were tested with the LLVM-based `amdflang` from `rocm-new/10.1.0` on AAC7 (or `rocm/10.1.0` on AAC6). Set up the environment accordingly.

The C++ version of this code is described in detail in this [ROCm blog](https://rocm.blogs.amd.com/high-performance-computing/jacobi/README.html).

The Fortran code is described in this [ROCm blog](https://rocm.blogs.amd.com/ecosystems-and-partners/fortran-journey/README.html).
```
cd 0_jacobi_portyourself
```
Decide if you want to port with or without unified shared memory. <hr>

**Either** for USM
```
export HSA_XNACK=1
```
<hr>

**or** without, for a behavior with copies
```
export HSA_XNACK=0
```
<hr>

Then set the compiler (e.g. `export FC=amdflang`, see the [README](../README.md) in the parent folder) and compile:
```
make
```
Run the code for a reference CPU solution.
```
./jacobi
```
You can specify a problem size with
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
Compare with your solution (e.g. use ```vimdiff``` to compare).
Note that instead of writing ```!$omp requires unified_shared_memory``` in each compilation unit we use the ```-fopenmp-force-usm``` flag in the Makefile. This saves the developer some typing and it allows the comparison of running same code with and without ```HSA_XNACK=1``` and with and without requiring unified shared memory. To switch the USM compilation off one can compile with ```make USM_FLAG=""```. This is especially interesting to compare the collected traces in timeline profiling where one can compare the traces with and without memory movement (see respective [rocprofv3](https://github.com/amd/HPCTrainingExamples/tree/main/Rocprofv3/OpenMP) and [rocprofiler-systems](https://github.com/amd/HPCTrainingExamples/tree/main/rocprofiler-systems/) exercises).

For the discrete GPU / manual memory management version:
```
export HSA_XNACK=0
cd 2_jacobi_targetdata
make
./jacobi
```
compare with your solution (e.g. use ```vimdiff``` to compare).

### Notes on enhancing this code with markers for profiling:
This is not part of the porting exercise but as you may like to experiment with profiling this example we included an example with ROCtx markers the profilers will show in the timelines and can use for some filtering options (see rocprofv3 and rocprofiler-systems exercises):
```
export HSA_XNACK=0
cd 3_jacobi_targetdata_markers
make
./jacobi
```
This version adds ROCtx ranges around selected code regions so that profiling
traces can show named regions. 

The example uses the local `roctx_mod` module to
declare the two C bindings needed in this exercise:

```
use roctx_mod, only: roctxRangePush, roctxRangePop
```
Using this small local module avoids a dependency on the Fortran hipfort
modules for the exercise. ROCm installations that provide hipfort also provide
these interfaces through the `hipfort` and `hipfort_roctx`
The marker version links the ROCtx library:
```
$(FC) $(LDFLAGS) $^ -o $@ -lrocprofiler-sdk-roctx
```
Rocprofv3 and Rocprofiler-systems exercises show how roctx markers show up in timeline tracing.
