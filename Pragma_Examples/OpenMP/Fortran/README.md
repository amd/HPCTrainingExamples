
# Porting exercises
The Fortran porting exercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
```
#### on aac6:

Follow the message of the day how to allocate a gpu interactively.
Load ROCm to set up the environment:
```
module load rocm
```

This module sets ```FC=amdflang``` for you.

#### on aac7:
Get an interactive session on a node:
```
salloc -N 1 --mem=100GB --gpus=1
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

You can choose the Cray Fortran compiler (ftn) or the LLVM-based amdflang compiler from ```rocm-new```.
##### LLVM-based amdflang (rocm-new) on aac7:
```
module load rocm-new
```
This module sets ```FC=amdflang``` for you, check with ```echo $FC```.

##### ftn compiler on aac7:
Prepare the environment (those modules should be default, check with ```module list```):
```
module load PrgEnv-cray
module load craype-accel-amd-gfx942
module swap rocm rocm-new/7.1.1
module load craype-accel-amd-gfx942
```
make sure to load the right rocm version (non-default!).
```
module list
```
should show you this list of modules:
```
Currently Loaded Modulefiles:
  1) craype-x86-genoa          7) cray-mpich/9.1.0
  2) libfabric/2.2.0rc1        8) cray-libsci/26.03.0
  3) craype-network-ofi        9) PrgEnv-cray/8.7.0
  4) perftools-base/26.03.0   10) rocm-new/7.1.1
  5) cce/21.0.0               11) craype-accel-amd-gfx942
  6) craype/2.7.36
```
export FC=ftn
```
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
The exercises 1-5 have a CPU only code to try porting yourself and (intermediate steps) of a solution. Exercise 6 does not have a version to port yourself, but explains a common challenge for porting to discrete GPUs.
The instructions assume you work on MI300A and some of the exercises explore the differences of using the discrete GPU or APU programming model (```HSA_XNACK=0``` or ```=1```).
The recommended order to do the exercises is the order in which they are numbered and first all with unified memory and then again with map clauses or data region.

Exercise 8 is a small app with a Jacobi solver. 
Note: A C/C++ version of this Fortran code is explained in detail in a blogpost 
https://rocm.blogs.amd.com/high-performance-computing/jacobi/README.html. 
The Fortran version is additionally described here: 
https://rocm.blogs.amd.com/ecosystems-and-partners/fortran-journey/README.html
<hr>

Choose one of the exercises in the sub-directories and use the README there for instructions, we reccomend to start with 1_saxpy:
```
cd 1_saxpy
cd 2_reduction  
cd 3_vecadd 
cd 4_reduction_scalars  
cd 5_reduction_array
cd 6_device_routines
cd 7_derived_types
cd 8_jacobi
```
There are also additional examples and tests in this folder to explore.
