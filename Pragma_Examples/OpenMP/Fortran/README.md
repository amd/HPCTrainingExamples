Porting excercises

cd $HOME/HPCTrainingExamples-Fortran/Pragma_Examples/OpenMP/Fortran

load the flang-new compiler and set up the environment 
module load rocm-afar-drop
export FC=flang-new

The exercises in the folders numbered 1 to 6 are small examples of what one may encounter when porting a real world code. 
cd 1_saxpy
cd 2_vecadd  
cd 3_freduce  
cd 4_reduction_scalar  
cd 5_device_routine 
cd 6_derived_types

Each excercise has it's own README with instructions.
The excercises 1-5 have a CPU only code to try porting yourself and (intermediate steps) of a solution. Excercise 6 does not have a version to port yourself, but explains a common challenge for porting to discrete GPUs.
The instructions assume you work on MI300A and some of the excercises explore the differences of using the discrete GPU or APU programming model (HSA_XNACK=0 or =1).
The reccomended order to do the exercises is the order in which they are numbered, but any sub-folder with excercises has instructions to do them stand-alone.
