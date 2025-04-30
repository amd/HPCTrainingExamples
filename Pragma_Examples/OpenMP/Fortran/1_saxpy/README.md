# First Fortran OpenMP offload: Porting saxpy step by step and explore the discrete GPU and APU programming models:

This is HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/1_saxpy/README.md in the training examples repository.

This excercise will show in a step by step solution how to port a your first kernels. 
This simple example will not use a Makefile to practice how to compile for the GPU or APU. 
All following excercises will use a Makefile.

There are 6 different enumerated folders. (Reccomendation: ```vimdiff saxpy.f90 ../<X_saxpy_version>/saxpy.f90``` may help you to see the differences):

First, prepare the environment (load modules, set environment variables), if you didn't do so before.

## Part 1: Porting with unified shared memory enabled
For now, set
```
export HSA_XNACK=1
```
to make use of the APU programming model (unified memory).
0) the serial CPU code.
```
cd 0_saxpy_serial_portyourself
```
Try to port this example yourself. If you are stuck, use the step by step solution in folders 1-6 and read the instructions for those excersices below. Recommendation for your first port: use ```!$omp requires unified_shared memory``` (in the code after ```implicit none``` in each module, can also be forced through a compiler flag ```-fopenmp-force-usm```) and ```export HSA_XNACK=1``` (before running) that you do not have to worry about map clauses. Steps 1-3 of the solution assume unified shared memory. Map clauses and investigating the behaviour of ```export HSA_XNACK=0``` or ```=1``` is added in the later steps.

- Compile the serial version. Note that ```-fopenmp``` is required as omp_get_wtime is used to time the loop execution.
```
amdflang -fopenmp saxpy.F90 -o saxpy
```
- Run the serial version.
```
./saxpy
```
You can now try to port the serial CPU version to the GPU or follow the
step by step solution:
1) Move the computation to the device
```
cd ../1_saxpy_omptarget
```
```
vi saxpy.f90
```
add ```!$omp target``` to move the loop in the saxpy subroutine to the device.
- Compile this first GPU version. Make sure you add ```--offload-arch=gfx942``` (on MI300A, find out what your system's gfx... is with ```rocminfo```)
on aac6 or aac7 with amdflang:
```
amdflang -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```
[Alternative: on systems with the Cray environment e.g. aac7 with cray ftn:

First, make sure you loaded the right module that offload is enabled before you compile with
```
ftn -fopenmp saxpy.F90 -o saxpy
```
]
- Run
```
./saxpy
```
The observed time is much larger than for the CPU version. More parallelism is required!

2) Add parallelism
```
cd ../2_saxpy_teamsdistribute
vi saxpy.f90
```
add "teams distribute"
- Compile again
- run again
The observed time is a bit better than in case 1 but still not the full parallelism is used.

3) Add multi-level parallelism
```
cd ../3_saxpy_paralleldosimd
vi saxpy.f90
``` 
add "parallel do" for more parellelism
- Compile again
- run again
The observed time is much better than all previous versions.
Note that the initialization kernel is a warm-up kernel here. If we do not have a warm-up kernel, the observed performance would be significantly worse. Hence the benefit of the accelerator is usually seen only after the first kernel. You can try this by commenting the !$omp target... in the initialize subroutine, then the meassured kernel is the first which touches the arrays used in the kernel.

## Part 2: explore the impact of unified shared memory
4) Explore impact of unified memory:
```
cd ../4_saxpy_nousm
vi saxpy.f90
```
The ```!$omp requires...``` line is removed.
- Compile again
- run again
so far we worked with unfied shared memory and the APU programming model. This allows good performance on MI300A, but not on discrete GPUs. In case you will work on discrete GPUs or want to write portable code for both discrete GPUs and APUs, you have to focus on data management, too.
```
export HSA_XNACK=0
```
to get similar behaviour like on discrete GPUs (with memory copies).
Compiling and running this version without any map clauses will result in much worse performance than with unified shared memory and ```HSA_XNACK=1``` (no memory copies on MI300A).

## Part 3: with map clauses
Set
```
export HSA_XNACK=0
```
that the map clauses do have an effect on MI300A.

5) this version introduces  map clauses for each kernel.
```
cd ../5_saxpy_map 
vi saxpy.f90
```
see where the map clasues where added. The x vector only has to be maped "to".
- compile again
- run again
The performance is not much better than version 4.

6) with enter and exit data clauses the memory is only moved once at the beginning the time to solution should be roughly in the order of magnitude of the unified shared memory version, but still slightly slower as the memory is copied like on discrete GPUs. Test yourself:
```
cd ../6_saxpy_targetdata
```
```
vi saxpy.f90
```
- compile again
- run again
Additional excercise: What happens to the result, if you comment the !$omp target update (in line 29)? 
```
vi saxpy.f90
```
- Don't forget to recompile after commenting it.

The results will be wrong! This shows, that proper validation of results is crutial when porting! Before you port a large app, think about your validation strategy before you start. Incremental testing is essential to capture such errors like missing data movement.

7) experiment with num_teams
```
cd ../7_saxpy_numteams
vi saxpy.f90
```
specify num_teams(...) choose a number of teams you want to test 
- compile again
- run again
investigating different numbers of teams you will find that the compiler default (without setting this) was already leading to good performance. saxpy is a very simple kernel, this finding may differ for very complex kernels.

After finishing this introductory excercise, go to the next excercise in the Fortran folder:
```
cd ../..
```

