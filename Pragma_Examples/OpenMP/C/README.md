
# Porting exercises
The C porting exercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples/Pragma_Examples/OpenMP/C
```
load the amdclang++  compiler and set up the environment

## on aac6: 
```
module load rocm
```
or
```
module load aomp/amdclang-19.0
```
```
export CC=amdclang
export CXX=amdclang++
```
## on aac7:
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

For amd compiler:
```
module load PrgEnv-amd
module load craype-accel-amd-gfx942
module load craype-x86-genoa
module load rocm
```
For cray compiler:
```
module load PrgEnv-cray
module load craype-accel-amd-gfx942
module load craype-x86-genoa
module load rocm
```
Check, if 
```
CC --version
```
shows a C compiler with offload capabilities.
Some Makefiles use the environment variable CXX, hence:
```
export CXX=CC
```
## Both systems:

This flag
```
export HSA_XNACK=1
```
will enable no memory copies (use of unified_shared_memory) on MI300A
```
export HSA_XNACK=0
```
will disable this and behave similar to a discrete GPU.
Check with
```
rocminfo
```
if xnack+ (unified memory enabled) or xnack- (with memory copies) is set.
# Excercises
The exercises in the folders numbered 1 to 6 are small examples of what one may encounter when porting a real world code. 
Each exercise has it's own README with instructions.
The exercises 1-6 have a CPU only code to try porting yourself and (intermediate steps) of a solution. Excercise 6 does not have a version to port yourself, but explains a common challenge for porting to discrete GPUs.
The instructions assume you work on MI300A and some of the exercises explore the differences of using the discrete GPU or APU programming model (HSA_XNACK=0 or =1).
The reccomended order to do the exercises is the order in which they are numbered, but any sub-folder with exercises has instructions to do them stand-alone.
Excercise 7 is a small app with a Jacobi solver. (Note: This code is explained in detail a blogpost https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-jacobi-readme/.) 

Choose one of the exercises in the sub-directories and use the README there for instructions (reccomended: follow them as they are numbered, first do each exercise with unified memory and later without):
```
cd 1_saxpy
cd 2_vecadd  
cd 3_reduction 
cd 4_reduction_scalars  
cd 5_reduction_array
cd 6_device_routine
```
