# Porting excercise device routine

This exercise will show how to port kernels which call a function.
Note: Make and build analogous to the previous excercises.
Each version has a sub-folder with a 
- serial CPU version to port yourself and a
- solution for unified memory and
- a asolution with map clauses.

Build and run analogous to the previous excercises.

There are three different versions:
```
cd 1_device_routine 
```
Explore the serial CPU code for each version (Hint: there are two files). 
How do you show the compiler to compile the function in the other file for the GPU?

```
cd 2_device_routine_wglobaldata  
```
Explore the serial CPU code for each version (Hint: there are two files). 
How do you show the compiler to compile the function in the other file for the GPU? How do you use the global data on the GPU?

```
cd 3_device_routine_wdynglobaldata
```
How do you show the compiler to compile the function in the other file for the GPU? How do you use the dynamic global data on the GPU?


