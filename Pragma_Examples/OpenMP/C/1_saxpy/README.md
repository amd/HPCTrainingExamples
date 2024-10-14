# First OpenMP offload: Porting saxpy step by step and explore the discrete GPU and APU programming models:

This excercise will show in a step by step solution how to port a your first kernels. 

# Excercise instructions:
For now, set
```
export HSA_XNACK=1
```
additionally, to make use of the APU programming model (unified memory).

There are 6 different enumerated folders. (Reccomendation: ```vimdiff saxpy.cpp ../<X_saxpy_version>/saxpy.cpp``` may help you to see the differences):

0) the serial CPU code.
```
cd 0_saxpy_serial_portyourself
```
Try to port this example yourself. If you are stuck, use the step by step solution in folders 1-6 and read the instructions for those excersices below. Recommendation for your first port: use ```#pragma omp requires unified_shared memory``` and ```export HSA_XNACK=1``` (before running) that you do not have to worry about map clauses. Steps 1-3 of the solution assume unified shared memory. Map clauses and investigating the behaviour of ```export HSA_XNACK=0``` or ```=1``` is added in the later steps.

- Compile the serial version. Note that ```-fopenmp``` is required as omp_get_wtime is used to time the loop execution.
```
amdclang++ -fopenmp saxpy.cpp -o saxpy
```
or with the cray environment (aac7):

```
CC -fopenmp saxpy.cpp -o saxpy
```

- Run the serial version.
```
./saxpy
```
Note: you can also use the Makefile.
```
make
```
instead of compiling manually.

You can now try to port the serial CPU version to the GPU 
```
vi saxpy.cpp
```
and don't forget to port the Makefile (Hint: What has to be added to compile for the GPU? Note: for cray compilers)
```
vi Makefile
```
or follow the step by step solution:
1) Move the computation to the device
```
cd ../1_saxpy_omptarget
```
```
vi saxpy.cpp
```
add ```#pragma omp target``` to move the loop in the saxpy subroutine to the device.
- Compile this first GPU version. Make sure you add ```--offload-arch=gfx942``` (on MI300A, find out what your system's gfx... is with ```rocminfo```)
```
amdclang++ -fopenmp --offload-arch=gfx942 saxpy.cpp -o saxpy
```
(or use the Makefile)
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
```
amdflang-new -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```
- run again
```
./saxpy
```
The observed time is a bit better than in case 1 but still not the full parallelism is used.

3) Add multi-level parallelism
```
cd ../3_saxpy_paralleldosimd
vi saxpy.f90
``` 
add "parallel do" for more parellelism
- Compile again
```
amdflang-new -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```
- run again
```
./saxpy
```
The observed time is much better than all previous versions.
Note that the initialization kernel is a warm-up kernel here. If we do not have a warm-up kernel, the observed performance would be significantly worse. Hence the benefit of the accelerator is usually seen only after the first kernel. You can try this by commenting the !$omp target... in the initialize subroutine, then the meassured kernel is the first which touches the arrays used in the kernel.

# Impact of USM
4) Explore impact of unified memory:
```
cd ../4_saxpy_nousm
vi saxpy.f90
```
The ```!$omp requires...``` line is removed.
- Compile again
```
amdflang-new -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```
- run again
```
./saxpy
 ```
so far we worked with unfied shared memory and the APU programming model. This allows good performance on MI300A, but not on discrete GPUs. In case you will work on discrete GPUs or want to write portable code for both discrete GPUs and APUs, you have to focus on data management, too.
```
export HSA_XNACK=0
```
to get similar behaviour like on discrete GPUs (with memory copies).
Compiling and running this version without any map clauses will result in much worse performance than with unified shared memory and ```HSA_XNACK=1``` (no memory copies on MI300A).

# Map clauses
5) this version introduces  map clauses for each kernel.
```
cd ../5_saxpy_map 
vi saxpy.f90
```
see where the map clasues where added. The x vector only has to be maped "to".
- Compile again
```
amdflang-new -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```
- run again
```
./saxpy
```
The performance is not much better than version 4.

6) with enter and exit data clauses the memory is only moved once at the beginning the time to solution should be roughly in the order of magnitude of the unified shared memory version, but still slightly slower as the memory is copied like on discrete GPUs. Test yourself:
```
cd ../6_saxpy_targetdata
```
```
vi saxpy.f90
```
- Compile again
```
amdflang-new -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```
- run again
```
./saxpy
```
Additional excercise: What happens to the result, if you comment the !$omp target update (in line 29)? 
```
vi saxpy.f90
```
Don't forget to recompile after commenting it.
```
amdflang-new -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```
The results will be wrong! This shows, that proper validation of results is crutial when porting! Before you port a large app, think about your validation strategy before you start. Incremental testing is essential to capture such errors like missing data movement.

7) experiment with num_teams
```
cd ../7_saxpy_numteams
vi saxpy.f90
```
specify num_teams(...) choose a number of teams you want to test 
- Compile again
```
amdflang-new -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```
- run again
```
./saxpy
```
investigating different numbers of teams you will find that the compiler default (without setting this) was already leading to good performance. Tuning num_teams or thread_limit may be required for some kernels, but the defaults are chosen quite well for saxpy. saxpy is a very simple kernel, this finding may differ for very complex kernels.

