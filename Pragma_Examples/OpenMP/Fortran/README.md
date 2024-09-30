# Porting excercises
The Fortran porting excercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples-Fortran/Pragma_Examples/OpenMP/Fortran
```
load the amdflang-new compiler and set up the environment 
```
module load amdflang-new-beta-drop
export FC=amdflang-new
```

The exercises in the folders numbered 1 to 6 are small examples of what one may encounter when porting a real world code. 
Each excercise has it's own README with instructions.
The excercises 1-5 have a CPU only code to try porting yourself and (intermediate steps) of a solution. Excercise 6 does not have a version to port yourself, but explains a common challenge for porting to discrete GPUs.
The instructions assume you work on MI300A and some of the excercises explore the differences of using the discrete GPU or APU programming model (HSA_XNACK=0 or =1).
The reccomended order to do the exercises is the order in which they are numbered, but any sub-folder with excercises has instructions to do them stand-alone.
Excercise 7 is a small app with a Jacobi solver. A C/C++ version of this Fortran version is explained in detail a Blogpost https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-jacobi-readme/.

```
cd 1_saxpy
cd 2_vecadd  
cd 3_freduce  
cd 4_reduction_scalar  
cd 5_device_routine 
cd 6_derived_types
cd 7_jacobi
```
