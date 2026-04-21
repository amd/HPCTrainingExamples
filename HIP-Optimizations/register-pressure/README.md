
## Register Exercises

In this set of examples, we explore

* VGPRs -- Vector General Purpose Registers
* SGPRs -- Scalar General Purpose Registers
* Occupancy

### Register Pressure - ROCm Blogs
For further reading on the details of the code versions explored here see: https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html
Note: Not all optimizations will work equally on all hardware and compiler versions and all types of code, but this exercise will show you different paths to reducing register pressure.

For these exercises, retrieve them with 

```
git clone https://github.com/AMD/HPCTrainingExamples
cd HPCTrainingExamples/rocm-blogs-codes/register-pressure
```

Set up your environment

```
module load rocm #or rocm-new depending on your system
```
Note: in CPE environment you can use```CC -x hip``` instead of ```hipcc```. This can be important for MPI applications that you link the correct libraries for MPI. This is not relevant for this excercise here, but something to keep in mind depending on the system you work on.

The exercises were tested on an MI210 with ROCm version 6.4.1 and MI300A with rocm 7.2.0.

Get the compiler resource report for the lbm.cpp kernel. Use the 
proper gfx model code in the compile command for <gfx-arch>.

```
hipcc -c --offload-arch=<gfx-arch> -Rpass-analysis=kernel-resource-usage lbm.cpp
```

Output should be something like
MI210 example:

```
lbm.cpp:16:1: remark:     SGPRs: 100 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     VGPRs: 104 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage
```
MI300A example:
```
lbm.cpp:16:1: remark: Function Name: _Z6kernelPdS_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_S_iiiiiiiddddddddddddddd [-Rpass-analysis=kernel-resource-usage]
   16 | {
      | ^
lbm.cpp:16:1: remark:     TotalSGPRs: 102 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     VGPRs: 102 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
lbm.cpp:16:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
```

Repeat for the other cases

#### Remove unnecessary math functions

pow(current_phi, 2.0) on line 37 can be changed to current_phi * current_phi

This C function raises the argument to a floating point power in software. It is not a very efficient way to do the
operation and also consumes a lot of registers.

```
hipcc -c --offload-arch=<gfx-arch> -Rpass-analysis=kernel-resource-usage lbm_1_nopow.cpp
```

#### Rearrange code so variables are declared close to use

```
hipcc -c --offload-arch=<gfx-arch> -Rpass-analysis=kernel-resource-usage lbm_2_rearrange.cpp
```

#### Add restrict attribute to function arguments

```
hipcc -c --offload-arch=<gfx-arch> -Rpass-analysis=kernel-resource-usage lbm_3_restrict.cpp
```

Try exploring other ways of reducing the number of VGPRs.

One way which might help is to use `__global__ __launch_bounds__(256) void kernel`
Try different workgroup sizes for launch bounds. Valid sizes would be 64, 128, 256, 512, and 1024.
Smaller should lead to fewer VGPRs.
 
