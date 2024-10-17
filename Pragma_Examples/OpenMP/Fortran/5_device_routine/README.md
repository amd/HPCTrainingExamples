# Porting excercise device routine and unified shared memory

This exercise will show how to port kernels which call a subroutine. (The same would apply to function calls.)
Note: Make and build analogous to the previous excercises.

## Part 1: Port the kernel with a subroutine call in an other file used using interface
```
cd device_routine_with_interface
```
there are five code versions in enumerated folders:
0) a serial CPU version. Try to port this yourself. Decide if you want to do a unified_shared_memory version (export HSA_XNACK=1) ore one with manual memory management (relevant for discrete GPUs). If you are stuck, explore solution versions in folders enumerated 1-4 (explanation below). Hint: Don't forget to adapt the Makefile and check, if your kernel really runs on the GPU.
#### Solution 1.1
1) a version which is only partially ported. It still results in an error! Do you know why? Hint: How does the compiler know to compile the routine in an other compilation unit (here an other file) for the GPU?

2) a solution version which runs on the GPU with unified shared memory

#### Solution 1.2
3) a solution version with optimized data movement (this is only relevant on discrete GPUs or with ```export HSA_XNACK=0```  on MI300A).


## Part 2: Port the kernel with a subroutine call in an other file using module
´´´
cd device_routine_with_module
´´´
there are three code versions in enumerated folders:
0) a serial CPU version. Try to port this yourself. 
Remember to 
```
export HSA_XNACK=1
```
for a unified_shared_memory version or

```
export HSA_XNACK=0
```
if you work on a version with memory copies. 
If you are stuck, explore solution versions in folders enumerated as 1 and 2 (explanation below). Hint: Don't forget to adapt the Makefile and check, if your kernel really runs on the GPU.

1) a solution with memory management (mapping) for discrete GPUS Note: this solution works for amdflang-new, with the cray compiler remove the link clause
2) a solution for unified shared memory  Note: this solution works for amdflang-new, with the cray compiler remove the link clause

After this excercise you should have learned the proper directives to compile a routine in an other compilation unit to be used in an OpenMP offloaded kernel, how to implement a reduction with openMP,  how to optimize the data management on discrete GPUs and how to benefit from unified_shared_memory on an APU.
