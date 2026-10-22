
# Porting exercises
The Fortran porting exercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
```
#### on aac6:

Follow the message of the day how to allocate a gpu interactively.
Load ROCm to set up the environment:
```
module load rocm/10.1.0
export FC=amdflang
```

Check with ```echo $FC``` and ```$FC --version``` that you are using `amdflang`.

#### on aac7:
Get an interactive session on a node:
```
salloc -N 1 --mem=100GB --gpus=1 -p 192C4G1H_MI300A_RHEL9_A1_CPX
```
Note: you will get 1 GPU and 100 GB of memory. This will allow others to use the remaining resources of a node.
Useful commands:
```
sinfo
```
check for available nodes.
```
squeue
```
check for your job(s). In case it was not terminated correctly, you may have to use
```
scancel <JobID>
```
to terminate a job.

Use the LLVM-based `amdflang` compiler from `rocm-new`:
```
module load rocm-new/10.1.0
export FC=amdflang
```
Check with ```echo $FC``` and ```$FC --version``` that you are using `amdflang`.
#### On all systems independent of the compiler:
This flag
```
export HSA_XNACK=1
```
will enable no memory copies (use of `unified_shared_memory`) on MI300A
```
export HSA_XNACK=0
```
will disable this and behave similarly to a discrete GPU with memory copies.
Check with
```
rocminfo
```
if ```xnack+``` (unified memory enabled) or ```xnack-``` (with memory copies) is set.

Note: If you don't want to add ```!$omp requires unified_shared_memory``` in every module by hand, you can use the compiler flag ```-fopenmp-force-usm```.

The exercises in the folders numbered 1 to 6 are small examples of what one may encounter when porting a real world code. 
Each exercise has its own README with instructions.
The exercises 1-6 have a CPU only code to try porting yourself and (intermediate steps) of a solution.
The instructions assume you work on MI300A and some of the exercises explore the differences of using the discrete GPU or APU programming model (```HSA_XNACK=0``` or ```=1```).
The recommended order to do the exercises is the order in which they are numbered and first all with unified memory and then again with map clauses or data region.

Exercise 8 is a small app with a Jacobi solver. 
Note: A C/C++ version of this Fortran code is explained in detail in a blogpost 
https://rocm.blogs.amd.com/high-performance-computing/jacobi/README.html. 
The Fortran version is additionally described here: 
https://rocm.blogs.amd.com/ecosystems-and-partners/fortran-journey/README.html
<hr>

Choose one of the exercises in the sub-directories and use the README there for instructions, we recommend to start with 1_saxpy. There are also additional examples and tests in this folder to explore.

## Exercise index

For new users of OpenMP offloading, we recommend following this order first: [saxpy](1_saxpy/), [reduction](2_reduction/), [device routines](6_device_routines/), [vecadd](3_vecadd/), and then [Jacobi](8_jacobi/). After that, use the other examples for special topics.

- [1_saxpy](1_saxpy/)
- [2_reduction](2_reduction/)
- [3_vecadd](3_vecadd/)
- [4_reduction_scalars](4_reduction_scalars/)
- [5_reduction_array](5_reduction_array/)
- [6_device_routines](6_device_routines/)
- [7_derived_types](7_derived_types/)
- [8_jacobi](8_jacobi/)
- [9_interop](9_interop/)
- [Atomic](Atomic/)
- [BuildExamples](BuildExamples/)
- [Common_blocks_on_device](Common_blocks_on_device/)
- [default_gotchas/declare_target](default_gotchas/declare_target/)
- [ComplexComputeConstructs](ComplexComputeConstructs/)
- [Nowait](Nowait/)
- [SingleLineConstructs](SingleLineConstructs/)
- [Submodules](Submodules/)
- [complex_saxpy](complex_saxpy/)
- [do_concurrent/1_saxpy](do_concurrent/1_saxpy/)
- [do_concurrent/2_reduction](do_concurrent/2_reduction/)
- [optimization/Allocations](optimization/Allocations/)
- [teams_endloop](teams_endloop/)
- [workdistribute](workdistribute/)

## Notes on OpenMP target offload exercises with the cray compiler

On systems with the Cray Programming Environment, the `ftn` wrapper can also be used. Load the matching ROCm and accelerator modules, for example `craype-accel-amd-gfx942`, and set `export FC=ftn`. The main exercise instructions use `amdflang` directly, so remember that direct compiler commands need `--offload-arch=gfx942` while Cray wrappers get the offload architecture from the loaded modules.
