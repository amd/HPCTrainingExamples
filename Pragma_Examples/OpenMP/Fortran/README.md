
# Porting exercises
The Fortran porting exercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
```
#### on aac6:

Follow the message of the day how to allocate a gpu interactively.
Load the amdflang-new compiler to set up the environment 
```
module load amdflang-new
```
The naming of the versions changed from drop 7.x to drop <llvm-version>.<drop_revision> e.g. afar-drop-22.2.0 to avoid confusion with rocm version numbers.

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

You can choose the Cray Fortran compiler (ftn) or the amdflang-new compiler.
##### amdflang-new compiler on aac7:
```
module load amdflang-new
```
This module sets ```FC=amdflang``` for you, check with ```echo $FC```.

##### ftn compiler on aac7:
Prepare the environment (those modules should be default, check with ```module list```):
```
module load PrgEnv-cray
module load craype-x86-genoa
module load craype-accel-amd-gfx942
module load cce
module load rocm
```
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

Note: In the beta release of the amdflang-new compiler ```HSA_XNACK=0``` with a code with ```!$omp requires unified_shared_memory``` can be compiled in some cases as if no ```unified_shared_memory``` is required. This is a behavior not according to the standard and will lead to an error message in future releases! Use the compiler flag ```-fopenmp-force-usm``` to enforce the correct behavior. This flag can also be used to enforce unified_shared_memory everywhere in the code compiled with it.

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
cd 2_vecadd  
cd 3_reduction 
cd 4_reduction_scalars  
cd 5_reduction_array
cd 6_device_routine
cd 7_derived_types
cd 8_jacobi
```
There are also additional examples and tests in this folder to explore.
