
# Porting exercises
The CXX porting exercises can be found here (this is the directory of this README): 
```
cd $HOME/HPCTrainingExamples/Pragma_Examples/OpenMP/CXX
```
#### on aac6: 

Follow the message of the day how to allocate one GPU interactively.
Load the amdclang compiler and set up the environment 
```
module load rocm
export CXX=amdclang++
```
#### on aac7:
Get an interactive session on a node:
```
salloc -N 1 --mem=100GB --gpus=1
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

You can choose the Cray C++ compiler (CC) or the amdclang++ compiler.
##### amdclang++ compiler on aac7:
Prepare the environment (should be default, check with ```module list```):
```
module load PrgEnv-amd
module load craype-x86-genoa
module load craype-accel-amd-gfx942
module load amd
module load rocm
```
```
export CXX=CC
```
Note that in CPE/25.03 the CC compiler wrapper in PrgEnv-amd leads to a segfault at program finalization. Therefore we decided to not recommend to use the compiler wrappers for now on aac7 with amdclang++. If you have rocm 6.3.3 or greater in that version of CPE you should not encounter any issues. You can use it without Cray wrappers by settimg:
```
module load rocm
```
```
export CXX=amdclang++
```
##### Cray C++ compiler on aac7:
Prepare the environment:
```
module load PrgEnv-cray
module load craype-x86-genoa
module load craype-accel-amd-gfx942
module load cce
module load rocm
```
```
export CXX=CC
```
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
