# Porting excercises
The Fortran porting excercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran
```
#### on aac6:

Load the amdflang-new compiler and set up the environment 
```
module load amdflang-new
export FC=amdflang-new
```
Note that everyone shares a single node, so performance can be severely impacted due to a noisy environment.

#### on aac7:
Get an interactive session on a node:
```
srun -N 1 --mem=100GB --gpus=1 --pty bash -i
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
check for you job(s). In case it was not terminated correctly, you may have to use
```
scancel <JobID>
```
to terminate a job.

You can choose the Cray Fortran compiler (ftn) or the amdflang-new compiler.
##### amdflang-new compiler on aac7:
```
module load rocm/rocm-afar-5891
```
```
export FC=amdflang-new
```
##### ftn compiler on aac7:
Prepare the environment:
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
#### on all systems independent of the compiler:
This flag
```
export HSA_XNACK=1
```
will enable no memory copies (use of unified_shared_memory) on MI300A
```
export HSA_XNACK=0
```
will disable this and behave similar to a discrete GPU with memory copies.
Check with
```
rocminfo
```
if xnack+ (unified memory enabled) or xnack- (with memory copies) is set.

Note: In the beta release of the amdflang-new/4.0 compiler ```HSA_XNACK=0``` with a code with !$omp requires unified_shared_memory can be compiled in some cases as if no unified_shared_memory is required. This is a behavior not according to the standard and will lead to an error message in future releases!

The exercises in the folders numbered 1 to 6 are small examples of what one may encounter when porting a real world code. 
Each excercise has it's own README with instructions.
The excercises 1-5 have a CPU only code to try porting yourself and (intermediate steps) of a solution. Excercise 6 does not have a version to port yourself, but explains a common challenge for porting to discrete GPUs.
The instructions assume you work on MI300A and some of the excercises explore the differences of using the discrete GPU or APU programming model (```HSA_XNACK=0``` or ```=1```).
The reccomended order to do the exercises is the order in which they are numbered and first all with unified memory and then again with map clauses or data region.
Excercise 7 is a small app with a Jacobi solver. (Note: A C/C++ version of this Fortran code is explained in detail a Blogpost https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-jacobi-readme/.) 

Choose one of the excercises in the sub-directories and use the README there for instructions (reccomended: follow them as they are numbered, do all excercises first with unified memory and then with map clauses):
```
cd 1_saxpy
cd 2_vecadd  
cd 3_reduction 
cd 4_reduction_scalars  
cd 5_device_routine 
cd 6_derived_types
cd 7_jacobi
```
