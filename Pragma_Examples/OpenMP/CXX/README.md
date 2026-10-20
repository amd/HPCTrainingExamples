
# Porting exercises
The CXX porting exercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples/Pragma_Examples/OpenMP/CXX
```
Note: it is reccomended to start with the C exercises first.
#### on aac6: 

Follow the message of the day how to allocate one GPU interactively.
```
salloc -N 1 --gpus=1 -p <partition>
```
```
module load rocm/10.1.0
export CXX=amdclang++
```
#### on aac7:
Get an interactive session on a node in the CPX queue:
```
salloc -N 1 --mem=20GB --gpus=1 -p 192C4G1H_MI300A_RHEL9_A1_CPX
```
Note: you will get 1 GPU and a fraction of the node memory. This will allow others to use the remaining resources of a node.

Use the `amdclang++` compiler from `rocm-new` module:
```
module load rocm-new/10.1.0
export CXX=amdclang++
```
#### Useful commands:
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

The exercises in the folders are small examples of what one may encounter when porting a real world code. 
Each exercise has its own README with instructions.
Exercise 8 is a small app with a Jacobi solver. (Note: This code is explained in detail a blogpost https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-jacobi-readme/.) 
Choose one of the exercises in the sub-directories and use the README there for instructions.

## Exercise index

For new users of OpenMP offloading, we recommend first following the introductory path in the C or Fortran sections: saxpy, reduction, device routines, vecadd, and then Jacobi. In this C++ section, [device routines](6_device_routines/) and [Jacobi](8_jacobi/) continue that path; the other examples cover special topics.

- [6_device_routines](6_device_routines/)
- [8_jacobi](8_jacobi/)
- [BuildExamples](BuildExamples/)
- [cpp_classes](cpp_classes/)
- [kernel_pragmas](kernel_pragmas/)
- [memory_pragmas](memory_pragmas/)
- [openmp-helloworld](openmp-helloworld/)
- [optimization/Allocations](optimization/Allocations/)
- [optimization/alignment](optimization/alignment/)

#### Notes on these exercises with the cray compiler

On systems with the Cray Programming Environment, the `CC` wrapper can also be used. Load the matching ROCm and accelerator modules, for example `craype-accel-amd-gfx942`, and set `export CXX=CC`. The main exercise instructions use `amdclang++` directly, so direct compiler commands need `--offload-arch=gfx942` while Cray wrappers get the offload architecture from the loaded modules.
