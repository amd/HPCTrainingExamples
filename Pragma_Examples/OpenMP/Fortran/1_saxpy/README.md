
## First Fortran OpenMP offload: Porting saxpy step by step and explore the discrete GPU and APU programming models:

This document is the README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/Fortran/1_saxpy` in the HPC Training Examples repository.

This exercise will guide you through porting your first kernels step by step by going through the 7 enumerated folders.

For this exercise, compile the code directly on the commandline for practicing how to compile code for the GPU or APU.
However, there is a Makefile provided in each folder and all following exercises will use Makefiles.

## Part 1: Porting with unified shared memory enabled
First, prepare the environment (load modules, set environment variables, check the instructions), if you didn't do so before.

For now, set
```
export HSA_XNACK=1
```
to make use of the APU programming model (unified memory).

Load a Fortran compiler.
We recommend to use the new LLVM-based `amdflang` Fortran compiler for these exercises, which ships as the default Fortran compiler since ROCm 7. Depending on the system you are on, you can simply load either
```
module unload rocm
module load rocm/10.1.0       # on AAC6
# or
module load rocm-new/10.1.0   # on AAC7

# Ensure Makefiles use correct compiler (in later exercises)
export FC=amdflang
```

Check that the correct compilers are loaded and the `FC` variable points to the correct compiler.
```
amdflang --version
$FC --version
```

> [!NOTE]
> Make sure to set the `FC` environment variable to the compiler you want to use, since this is how the Makefiles in this and later exercises will choose the compiler to use.

### 1.0) the serial CPU code.
```
cd 0_saxpy_portyourself
```

Compile the serial version. Note that ```-fopenmp``` is required as ```omp_get_wtime``` is used to time the loop execution.
```
amdflang -fopenmp saxpy.F90 -o saxpy
```

Run the serial version.
```
./saxpy
```

The expected output is `y(1)4.000000` and `y(n-1)4.000000`.
You can now try to port the serial CPU version to the GPU or follow the
step by step sample solution and ideas in the following.
Suggestion: Use ```vimdiff``` to compare the different versions.

Hints: use ```!$omp requires unified_shared_memory``` (in the code after ```implicit none``` in each module or compiler flag ```-fopenmp-force-usm```) and ```export HSA_XNACK=1``` (before running). Steps 1-3 of the solution assume unified shared memory. Map clauses and investigating the behavior of ```export HSA_XNACK=0``` or ```=1``` is added in the later steps. Make sure to set ```--offload-arch``` for compilation!

### 1.1) Partial insufficient solution: First step: move the computation to the device
```
cd ../1_saxpy_omptarget
```
```
vi saxpy.F90
```
adds ```!$omp target``` to move the loop in the saxpy subroutine to the device.

Compile this first GPU version.
Make sure you add ```--offload-arch=gfx942``` (on MI300A, find out what your system's gfx is, e.g. with ```rocminfo | grep gfx```)
with the `amdflang` compiler.
```
amdflang -fopenmp --offload-arch=gfx942 saxpy.F90 -o saxpy
```

Run
```
./saxpy
```

The observed time is much larger than for the CPU version which shows: More parallelism is required to make use of the GPU!

### 1.2) Partial insufficient solution: Add parallelism
```
cd ../2_saxpy_omptargetparalleldo
vi saxpy.F90
```
add ```parallel do```
- compile again
- run again

The observed time is a bit better than in case 1.1 but still not the full parallelism is used.

### 1.3) Solution: Add multi-level parallelism to use the full device
```
cd ../3_saxpy_fullpragma
```
and inspect
```
vi saxpy.F90
```
Adds `teams distribute` to the directive (`target teams distribute parallel do`) for more parallelism.
- compile again
- run again

The observed time is much better than all previous versions.
Note that the initialization kernel is a warm-up kernel here. If we do not have a warm-up kernel, the observed performance would be significantly worse. Hence the benefit of the accelerator is usually seen only after the first kernel touching the data on the device when system allocators were used. You can try this by commenting out the ```!$omp target...``` in the initialize subroutine, then the measured kernel is the first which touches the arrays used in the kernel. A way to circumvent the penalty is using ```omp_target_alloc``` if the data is only needed on the device.

Note: you could also switch around 1.2 and 1.3.

### 1.4) Experiment with different versions
Check the output of different combinations of directives. What can you learn from setting the following environment variables before running?

For the Next Generation Fortran compiler check:
- Set ```export LIBOMPTARGET_KERNEL_TRACE=1``` (or 2 or 3 or -1 for all info). Can you interpret the output? Set it back to zero afterwards.
to get rid of the verbose output for subsequent runs later ```unset LIBOMPTARGET_KERNEL_TRACE```
- Set ```export LIBOMPTARGET_INFO=1``` (or 2 or 3 or -1 for all info). Can you interpret the output? Set it back to zero afterwards.
to get rid of the verbose output for subsequent runs later ```unset LIBOMPTARGET_INFO```

## Part 2: explore the impact of unified shared memory
```
cd ../4_saxpy_nousm
vi saxpy.F90
```
The ```!$omp requires...``` line is removed.
- compile again
- run again

So far, we worked with unified shared memory and the APU programming model. This allows good performance on MI300A, but not on discrete GPUs. In case you will work on discrete GPUs or want to write portable code for both discrete GPUs and APUs, you have to focus on data management, too. Set
```
export HSA_XNACK=0
```
to get similar behavior like on discrete GPUs (with memory copies). You can repeat exercise 1.4 with both versions and compare: You will see additional data movement.
Compiling and running this version without any map clauses but with memory copies will result in much worse performance than with unified shared memory and ```HSA_XNACK=1``` (no memory copies on MI300A).

Note: If you don't want to add ```!$omp requires unified_shared_memory``` in every module by hand, you can use the compiler flag ```-fopenmp-force-usm```.

## Part 3: with map clauses
Set
```
export HSA_XNACK=0
```
for the map clauses to have an effect on MI300A. You can try on your own to port the unified memory version from Part 1 or follow the sample solution.

### 3.1) introduce map clauses for each kernel
```
cd ../5_saxpy_map
vi saxpy.F90
```
See where the map clauses were added. The ```x``` vector only has to be mapped "to".
- compile again
- run again

The performance is not much better than version 4 (`4_saxpy_nousm`).

### 3.2) with enter and exit data clauses
The memory is only moved once at the beginning. The time to solution should be roughly in the order of magnitude of the unified shared memory version, but still slightly slower as the memory is copied once like on discrete GPUs. Test yourself:
```
cd ../6_saxpy_targetdata
```
```
vi saxpy.F90
```
- compile again
- run again

Additional exercise: What happens to the result, if you comment out the ```!$omp target update``` (in line 31)?
```
vi saxpy.F90
```
Don't forget to recompile after commenting it out.
The results will be wrong! This shows that proper validation of results is crucial when porting! Before you port a large app, think about your validation strategy before you start. Incremental testing is essential to capture such errors like missing data movement.

### 3.3) Investigate
What can you learn from setting those environment variables?

For the Next Generation Fortran compiler:
Set
```
export LIBOMPTARGET_KERNEL_TRACE=1
```
(or 2 or 3 or -1 for all info). Then run the app again. Can you interpret the output? Set it back to zero afterwards.
Try the same for
```
export LIBOMPTARGET_INFO=1
```
(or 2 or 3 or -1 for all info). Can you interpret the output? Set it back to zero afterwards.

What are the differences compared to the USM version?

### 3.4) adding an additional clause
Experiment with the ```num_teams``` clause.
```
cd ../7_saxpy_numteams
vi saxpy.F90
```
Specify ```num_teams(...)``` and choose a number of teams you want to test.
- compile again
- run again

When investigating different numbers of teams you will find that the compiler default (without setting this) was already leading to good performance (with recent rocm versions, older rocm versions may profit from more help). saxpy is a very simple kernel, this finding may differ for complex kernels with many instructions.
Note: ```num_teams``` is just one example how to control more what the compiler does. There are several other clauses which can be combined with ```target teams distribute parallel do```. <hr>

After finishing this introductory exercise, go to the next exercise in the Fortran folder:
```
cd ../..
```

