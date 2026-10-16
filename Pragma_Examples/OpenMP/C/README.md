
# Porting exercises
The C porting exercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples/Pragma_Examples/OpenMP/C
```
#### on aac6: 

Follow the message of the day how to allocate one GPU interactively.
Load the amdclang compiler and set up the environment 
```
module load rocm/10.1.0
export CC=amdclang
export C_COMPILER=amdclang
```
#### on aac7:
Get an interactive session on a node:
```
salloc -N 1 --mem=100GB --gpus=1 -p 192C4G1H_MI300A_RHEL9_A1_CPX
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

Use the `amdclang` compiler from `rocm-new`:
```
module load rocm-new/10.1.0
export CC=amdclang
export C_COMPILER=amdclang
```
> [!NOTE]
> Some C examples use `C_COMPILER` instead of only `CC` because `CC` is also the Cray C++ compiler wrapper; CMake and a few Makefiles use the standard `CC` variable.
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

The exercises in the folders numbered 1 to 6 are small examples of what one may encounter when porting a real world code. 
Each exercise has its own README with instructions.
The exercises 1-5 have a CPU only code to try porting yourself and (intermediate steps) of a solution. Exercise 6 does not have a version to port yourself, but explains a common challenge for porting to discrete GPUs.
The instructions assume you work on MI300A and some of the exercises explore the differences of using the discrete GPU or APU programming model (```HSA_XNACK=0``` or ```=1```).
The recommended order to do the exercises is the order in which they are numbered and first all with unified memory and then again with map clauses or data region.

Exercise 8 is a small app with a Jacobi solver that you can find in the CXX section. (Note: This code is explained in detail a blogpost https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-jacobi-readme/.)

## Exercise index

For new users of OpenMP offloading, we recommend following this order first: [saxpy](1_saxpy/), [reduction](2_reduction/), [device routines](6_device_routines/), [vecadd](3_vecadd/), and then the [Jacobi solver in the C++ section](../CXX/8_jacobi/). After that, use the other examples for special topics.

- [1_saxpy](1_saxpy/)
- [2_reduction](2_reduction/)
- [3_vecadd](3_vecadd/)
- [4_reduction_scalars](4_reduction_scalars/)
- [5_reduction_array](5_reduction_array/)
- [6_device_routines](6_device_routines/)
- [Atomics](Atomics/)
- [BuildExamples](BuildExamples/)
- [ComplexComputeConstructs](ComplexComputeConstructs/)
- [default_gotchas/declare_target](default_gotchas/declare_target/)
- [Nowait](Nowait/)
- [SingleLineConstructs](SingleLineConstructs/)
- [optimization/Allocations](optimization/Allocations/)

#### Notes on this exercise with the cray compiler

On systems with the Cray Programming Environment, the `cc` wrapper can also be used. Load the matching ROCm and accelerator modules, for example `craype-accel-amd-gfx942`, and set `export C_COMPILER=cc`. The main exercise instructions use `amdclang` directly, so direct compiler commands need `--offload-arch=gfx942` while Cray wrappers get the offload architecture from the loaded modules.
