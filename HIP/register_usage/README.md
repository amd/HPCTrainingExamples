# See how kernel implementation affects scalar and vector registers usage

This example is aimed at showing how different kernel implementations can affect the general purpose registers usage.
Registers are memory resources available at the compute unit level, i.e. each compute unit has access to a certain number of scalar general purpose registers (SGPRs) and vector general purpose registers (VGPRs). The number of registers of each type depends on your hardware. The SGPRs are used to store data that is uniform across the threads, i.e. that does not depend on the specific thread. VGPRs are used to store data that is different across threads, for instance data that depends on the global thread ID.

In this example, we will see two implementations of a kernel that intentionally makes heavy use of registers. The example is meant to show programmers the impact that the implementation of your kernel can have on the use of registers. Remember that, unlike LDS, registers are not programmable and the compiler is in charge of deciding how to use them. We also want to point out that what emphasis has to be put on how the kernel performs the computations rather than what it is doing, which is intentionally nothing particularly meaningful or physically relevant.

## Kernel heavily using SGPRs

If you inspect `register_usage.hip`, you'll see a kernel called `register_heavy_kernel` that is taking in a template parameter called `REG_PRESSURE`: in the main program there are subsequent calls to the kernel where the value of `REG_PRESSURE` is progressively doubled. The `Makefile` is written so that by default the implementation of `register_heavy_kernel` selected is the one that heavily using SGPRs. Inspecting the compilation output, it can be seen that as the value of `REG_PRESSURE` is increased, more SGPRs are used, but the number of VGPRs remains the same. This is intentional to provide an example of a kernel that is specifically making heavy use of SGPRs. 

To compile and run do:
```
module load rocm
make
./register_usage
```

The compilation output should look similar to this:

```
register_usage.hip:15:1: remark: Function Name: _Z21register_heavy_kernelILi8EEvPKiPiiiiii [-Rpass-analysis=kernel-resource-usage]
   15 |     const int* input, int* output, int param1, int param2, int param3, int param4, int n){
      | ^
register_usage.hip:15:1: remark:     TotalSGPRs: 18 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs: 4 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark: Function Name: _Z21register_heavy_kernelILi16EEvPKiPiiiiii [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     TotalSGPRs: 27 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs: 4 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark: Function Name: _Z21register_heavy_kernelILi32EEvPKiPiiiiii [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     TotalSGPRs: 46 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs: 4 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark: Function Name: _Z21register_heavy_kernelILi64EEvPKiPiiiiii [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     TotalSGPRs: 99 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs: 4 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
/shared/apps/rhel9/rocm-7.2.0/bin/hipcc register_usage.o  -o register_usage
```

## Kernel heavily using VGPRs

To use the version of th `register_heavy_kernel` that is making heavy use of VGPRs, compile supplying the `VGPR=1` flag:

```
module load rocm
make VGPR=1
./register_usage
```

Inspecting the compilation output, you will see that as the number of SGPRs remains the same, the number of VGPRs increases. The output should look something like this:

```
register_usage.hip:15:1: remark: Function Name: _Z21register_heavy_kernelILi8EEvPKiPiiiiii [-Rpass-analysis=kernel-resource-usage]
   15 |     const int* input, int* output, int param1, int param2, int param3, int param4, int n){
      | ^
register_usage.hip:15:1: remark:     TotalSGPRs: 12 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs: 21 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark: Function Name: _Z21register_heavy_kernelILi16EEvPKiPiiiiii [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     TotalSGPRs: 12 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs: 41 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Occupancy [waves/SIMD]: 8 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark: Function Name: _Z21register_heavy_kernelILi32EEvPKiPiiiiii [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     TotalSGPRs: 12 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs: 70 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Occupancy [waves/SIMD]: 7 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark: Function Name: _Z21register_heavy_kernelILi64EEvPKiPiiiiii [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     TotalSGPRs: 12 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs: 79 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     AGPRs: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     ScratchSize [bytes/lane]: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Dynamic Stack: False [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     Occupancy [waves/SIMD]: 6 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     SGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     VGPRs Spill: 0 [-Rpass-analysis=kernel-resource-usage]
register_usage.hip:15:1: remark:     LDS Size [bytes/block]: 0 [-Rpass-analysis=kernel-resource-usage]
/shared/apps/rhel9/rocm-7.2.0/bin/hipcc register_usage.o  -o register_usage
```

If you have a kernel in your application that is making heavy use of registers, you may be increasing your register pressure to a point where occupancy may be reduced as a result, limiting the performance of your kernel. See the [register pressure](https://rocm.blogs.amd.com/software-tools-optimization/register-pressure/README.html) blog post on how to recognize excessive register pressure and what to do to reduce it.
