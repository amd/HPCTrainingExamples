# Register Exercises

In this set of examples, we explore

* VGPRs -- Vector General Purpose Registers
* SGPRs -- Scalar General Purpose Registers
* Occupancy

For these exercises, retrieve them with 

```
git clone https://github.com/AMD/HPCTrainingExamples
cd HPCTrainingExamples/rocm-blogs-codes/registerpressure
```

Set up your environment

```
module load rocm
```

The exercises were tested on an MI210 with ROCm version 6.4.1.

Get the compiler resource report for the lbm.cpp kernel. Use the 
proper gfx model code in the compile command.

```
hipcc -c --offload-arch=gfx90a -Rpass-analysis=kernel-resource-usage lbm.cpp
```

Output should be something like

```
lbm.cpp:16:1: remark:     SGPRs: 100 [-Rpass-analysis=kernel-resource-usage]​
lbm.cpp:16:1: remark:     VGPRs: 104 [-Rpass-analysis=kernel-resource-usage]​
lbm.cpp:16:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]​
lbm.cpp:16:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]​
lbm.cpp:16:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]​
lbm.cpp:16:1: remark:     Occupancy [waves/SIMD]: 4 [-Rpass-analysis=kernel-resource-usage]​
lbm.cpp:16:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]​
lbm.cpp:16:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]​
lbm.cpp:16:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage​
```

Repeat for the other cases

1. Remove unnecessary math functions​

pow(current_phi, 2.0) on line 37 can be changed to current_phi * current_phi​

2. This C function raises the argument to a floating point power in software. It is not a very efficient way to do the
operation and also consumes a lot of registers.​

3. Rearrange code so variables are declared close to use​

Add restrict attribute to function arguments

```
hipcc -c --offload-arch=gfx90a -Rpass-analysis=kernel-resource-usage lbm_1_nopow.cpp
hipcc -c --offload-arch=gfx90a -Rpass-analysis=kernel-resource-usage lbm_2_rearrange.cpp
hipcc -c --offload-arch=gfx90a -Rpass-analysis=kernel-resource-usage lbm_3_restrict.cpp
```

Try exploring other ways of reducing the number of VGPRs.

One way which might help is to use `__global__ __launch_bounds__(256) void kernel`
Try different workgroup sizes for launch bounds. Valid sizes would be 64, 128, 256, 512, and 1024.
Smaller should lead to fewer VGPRs.
 
