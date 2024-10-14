# Porting excercises
The C porting excercises can be found here (this is the directory of this README): 
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

```
module load PrgEnv-amd
module load craype-accel-amd-gfx942
module load craype-x86-genoa
module load rocm
```
Check, if 
```
CC --version
```
shows a C compiler with offload capabilities.
```
export CXX=amdclang++
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

# Excercises
The exercises in the folders numbered 1 to 6 are small examples of what one may encounter when porting a real world code. 
Each excercise has it's own README with instructions.
The excercises 1-6 have a CPU only code to try porting yourself and (intermediate steps) of a solution. Excercise 6 does not have a version to port yourself, but explains a common challenge for porting to discrete GPUs.
The instructions assume you work on MI300A and some of the excercises explore the differences of using the discrete GPU or APU programming model (HSA_XNACK=0 or =1).
The reccomended order to do the exercises is the order in which they are numbered, but any sub-folder with excercises has instructions to do them stand-alone.
Excercise 7 is a small app with a Jacobi solver. (Note: This code is explained in detail a Blogpost https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-jacobi-readme/.) 

Choose one of the excercises in the sub-directories and use the README there for instructions (reccomended: follow them as they are numbered):
```
cd 1_saxpy
cd 2_vecadd  
cd 3_reduction 
cd 4_reduction_scalars  
cd 5_reduction_array
cd 6_device_routine
cd 7_jacobi
```
