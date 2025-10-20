
## First C OpenMP offload: Porting saxpy step by step and explore the discrete GPU and APU programming models:

This document is the README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/C/1_saxpy` in the HPC Training Examples repository.

This exercise will show in a step by step solution how to port your first kernels. 
This simple example will practice how to compile for the GPU or APU on the commandline and not use Makefiles. However, there is a Makefile in each folder you can use. All following exercises will use a Makefile.

There are 7 different enumerated folders. The following exercises will guide you step by step:

## Part 1: Porting with unified shared memory enabled
First, prepare the environment (load modules, set environment variables), if you didn't do so before.

For now, set
```
export HSA_XNACK=1
```
to make use of the APU programming model (unified memory).

Load a C compiler module <hr>
**either**
The AMD Compiler
```
module load PrgEnv-amd
module load rocm
module load amd
export C_COMPILER = amdclang
```
Note: to correctly use the cray wrappers one should export C_COMPILER = cc, as the cray wrappers are broken in combination with rocm 6.3.0 we reccomend using amdclang directly in this training. This is not a good advise in general if you e.g. want to link Cray MPICH in your application!

**or** (only available on systems with CPE installed)
```
module load PrgEnv-cray
module load rocm
module load cce
export C_COMPILER = cc
```

Check with 
```
$C_COMPILER --version
```
if you loaded the compiler you expect.
> [!NOTE]
> Setting ```C_COMPILER``` to the desired compiler is only necessary when using the provided Makefiles in this or later examples. It is not required in this example if you compile by hand as shown in the following for educational purposes.
> The `C_COMPILER` environment variable is chosen instead of the standard `CC` variable for C compilers since `CC` clashes with the Cray C++ compiler wrapper `CC`.

#### 1.0) the serial CPU code.
```
cd 0_saxpy_portyourself
```
Try to port this example yourself. If you are stuck, use the step by step solution in folders 1-7 and read the instructions for those exersices below. Recommendation for your first port: use ```#pragma omp requires unified_shared memory``` and ```export HSA_XNACK=1``` (before running) so that you do not have to worry about map clauses. Steps 1-3 of the solution assume unified shared memory. Map clauses and investigating the behaviour of ```export HSA_XNACK=0``` or ```=1``` is added in the later steps.

Compile the serial version. Note that ```-fopenmp``` is required as ```omp_get_wtime``` is used to time the loop execution.<hr>
**either**
```
amdclang -fopenmp saxpy.c -o saxpy
```
 <hr>
 
**or**
```
cc -fopenmp saxpy.c -o saxpy
```
<hr>

Run the serial version.
```
./saxpy
```
You can now try to port the serial CPU version to the GPU or follow the
step by step solution and ideas:

### 1.1) Move the computation to the device
```
cd ../1_saxpy_omptarget
```
```
vi saxpy.c
```
add ```#pragma omp target``` to move the loop in the saxpy subroutine to the device.

Compile this first GPU version.<hr>
**either**
Make sure you add ```--offload-arch=gfx942``` (on MI300A, find out what your system's gfx... is with ```rocminfo```)
```
amdclang -fopenmp --offload-arch=gfx942 saxpy.c -o saxpy
```
 <hr>

**or** on systems with CPE with cc:

First, make sure you loaded the right module that offload is enabled ```module load craype-accel-amd-gfx942``` (for MI300A) before you compile with
```
cc -fopenmp saxpy.c -o saxpy
```
Note that CPE compiler wrapper ```cc``` determines the offload architecture through modules while using compilers directly requires you to add the ```--offload-arch``` to the compilation line (see above).
<hr>

Run
```
./saxpy
```

The observed time is much larger than for the CPU version which shows: More parallelism is required to make use of the GPU!

### 1.2) Add parallelism
```
cd ../2_saxpy_teamsdistribute
vi saxpy.c
```
add ```teams distribute```
- Compile again
- run again
The observed time is a bit better than in case 1.1 but still not the full parallelism is used.

### 1.3) Add multi-level parallelism
```
cd ../3_saxpy_parallelforsimd
vi saxpy.c
``` 
Add "parallel for" for more parellelism.
- Compile again
- run again
The observed time is much better than all previous versions.
Note that the initialization kernel is a warm-up kernel here. If we do not have a warm-up kernel, the observed performance would be significantly worse. Hence the benefit of the accelerator is usually seen only after the first kernel. You can try this by commenting the ```#pragma omp target...``` in the initialize subroutine, then the measured kernel is the first which touches the arrays used in the kernel. A way to circumvent the penalty is using ```omp_target_alloc``` if the data is only needed on the device.

Note: you could also switch around 1.2 and 1.3.

### 1.4) Experiment with different versions
Check the output of different combinations of directives. What can you learn from setting those environment variables?
 <hr>
 
For the Next Generation Fortran compiler check:
- Set ```export LIBOMPTARGET_KERNEL_TRACE=1``` (or 2 or 3 or -1 for all info). Can you interpret the output? Set it back to zero afterwards.
- Set ```export LIBOMPTARGET_INFO=1``` (or 2 or 3 or -1 for all info). Can you interpret the output? Set it back to zero afterwards.
 <hr>
 
For the Cray Fortran compiler:
Set ```export CRAY_ACC_DEBUG=1``` (or 2 or 3). Can you interpret the output? Set it back to zero afterwards.
 <hr>

## Part 2: explore the impact of unified memory
```
cd ../4_saxpy_nousm
vi saxpy.c
```
The ```#pragma omp requires...``` line is removed.
- Compile again
- run again
So far, we worked with unified shared memory and the APU programming model. This allows good performance on MI300A, but not on discrete GPUs. In case you will work on discrete GPUs or want to write portable code for both discrete GPUs and APUs, you have to focus on data management, too. Set
```
export HSA_XNACK=0
```
to get similar behaviour like on discrete GPUs (with memory copies). You can repeat exercise 1.4 with both versions and compare: You will see additional data movement.
Compiling and running this version without any map clauses but with memory copies will result in much worse performance than with unified shared memory and ```HSA_XNACK=1``` (no memory copies on MI300A).

## Part 3: with map clauses
Set
```
export HSA_XNACK=0
```
for the map clauses to have an effect on MI300A.

### 3.1) introduce map clauses for each kernel
```
cd ../5_saxpy_map 
vi saxpy.c
```
See where the map clasues where added. The ```x``` vector only has to be maped "to".
- compile again
- run again
The performance is not much better than version 4.

### 3.2) with enter and exit data clauses 
The memory is only moved once at the beginning. The time to solution should be roughly in the order of magnitude of the unified shared memory version, but still slightly slower as the memory is copied once like on discrete GPUs. Test yourself:
```
cd ../6_saxpy_targetdata
```
```
vi saxpy.c
```
- compile again
- run again
Additional exercise: What happens to the result, if you comment out the ```#pragma omp target update```? 
```
vi saxpy.c
```
Don't forget to recompile after commenting it.
- Don't forget to recompile after commenting it.
The results will be wrong! This shows, that proper validation of results is crucial when porting! Before you port a large app, think about your validation strategy before you start. Incremental testing is essential to capture such errors like missing data movement.

> [!TIP]
> In C++, you can use the ```new``` allocator with an alignment of 128 instead of ```malloc``` to control the memory alignment. This is beneficial for improved performance:
> ```cpp
> #include <memory>
> // ...
> x = new (std::align_val_t(128)) float[N];
> y = new (std::align_val_t(128)) float[N];
> ```

### 3.3) Investigate
What can you learn from setting those environment variables?
 <hr>
 
For the amdclang compiler:
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
 <hr>
 
For the Cray C compiler:
Set 
```
export CRAY_ACC_DEBUG=1
``` 
(or 2 or 3). Can you interpret the output? Set it back to zero afterwards.<hr>

What are the differences compared to the USM version?

### 3.4) adding an additional clause
Experiment with the ```num_teams``` clause.
```
cd ../7_saxpy_numteams
vi saxpy.c
```
Specify ```num_teams(...)```` and choose a number of teams you want to test.
- compile again
- run again
When investigating different numbers of teams you will find that the compiler default (without setting this) was already leading to good performance (with recent rocm versions, older rocm versions may profit from more help). saxpy is a very simple kernel, this finding may differ for complex kernels with many instructions.
Note: ```num_teams``` is just one example how to control more what the compiler does. There are several other clauses which can be combined with ```target teams distribute parallel for```. <hr>

After finishing this introductory exercise, go to the next exercise in the C folder:
```
cd ../..
```
