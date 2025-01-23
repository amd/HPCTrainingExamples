# RocProfv3

One of the easiest tools to identify kernel hotspots in an application is rocprofv3, available in ROCm 6.2.0 and onward.  As a straightforward tracing tool, rocprofv3 can perform sample based profiling of applications and is compatible with applications at scale, producing one output per process.  An example of using rocprofv3 via slurm  is available with this document.  The relevant command for launching a rocprofv3 command is:

```
rocprofv3 --stats --sys-trace --kernel-trace -- python3 <application>.py
```

An example of this is shown in `kernels.sh`.  By default, the output format is csv format, which will produce a number of files per process.  For example, with the configuration above you will be provided with the following outputs:

```bash
> ls 211*
2117183_agent_info.csv  2117183_domain_stats.csv  2117183_kernel_stats.csv  2117183_kernel_trace.csv
```

At a high level view, the kernel_stats.csv file may be most useful for identifying which kernels are of particular interest.  For the run above, the top kernels in the application (which is not necessarily an optimized application) are:

```bash
head 2117183_kernel_stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"__amd_rocclr_copyBuffer",7084,17229993,2432.240683,8.38,601,10656,1241.386133
"MIOpenBatchNormBwdSpatial",1113,13283773,11935.106020,6.46,6009,51117,6712.148824
"void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::Array<char*, 2> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::Array<char*, 2>)",3402,12535732,3684.812463,6.10,641,12458,1127.090878
"MIOpenBatchNormFwdTrainSpatial",1113,10454471,9393.055705,5.08,5048,20351,3723.028735
"SubTensorOpWithScalar1d",2898,10330053,3564.545549,5.02,1161,16345,1668.462768
"igemm_fwd_gtcx3_nhwc_fp16_bx0_ex0_bt128x128x32_wt32x32x8_ws1x1_wr2x2_ta1x8x2x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs",609,9725160,15969.064039,4.73,11818,29164,3533.613567
"igemm_wrw_gtcx3_nhwc_fp16_bx0_ex1_bt256x256x32_wt32x32x8_ws2x2_wr2x2_ta1x4x1x8_1x8x1x32_tb1x4x1x8_1x8x1x32_gkgs",252,7631141,30282.305556,3.71,24877,43866,2466.553834
"igemm_wrw_gtcx3_nhwc_fp16_bx0_ex0_bt256x256x32_wt32x32x8_ws2x2_wr2x2_ta1x4x1x8_1x8x1x32_tb1x4x1x8_1x8x1x32_gkgs",378,7155943,18931.066138,3.48,13741,31607,4078.800286
"igemm_bwd_gtcx3_nhwc_fp16_bx0_ex0_bt128x128x32_wt32x32x8_ws1x1_wr2x2_ta1x8x2x1_1x4x1x64_tb1x8x1x2_1x4x1x64_gkgs",462,7049063,15257.712121,3.43,11938,27081,2561.401809> 
```

Let's compare that to the top kernels in a parallelized run, from `slurm_kernels.sh` or `mpi_kernels.sh`, depending on your system:

```bash
>  head 2118357_kernel_stats.csv
"Name","Calls","TotalDurationNs","AverageNs","Percentage","MinNs","MaxNs","StdDev"
"rccl_main_kernel(ncclDevComm*, unsigned long, ncclWork*)",142,916515855,6454337.007042,82.74,7210,382428559,32102362.228032
"__amd_rocclr_copyBuffer",6768,16006424,2365.015366,1.44,601,13140,1789.526108
"MIOpenBatchNormBwdSpatial",1060,12805350,12080.518868,1.16,4847,54241,6935.866504
"void at::native::vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::Array<char*, 2> >(int, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail::Array<char*, 2>)",3220,10130050,3145.978261,0.9145,681,13781,1055.132363
"MIOpenBatchNormFwdTrainSpatial",1060,9491028,8953.800000,0.8568,3806,18106,3643.596736
"igemm_fwd_gtcx3_nhwc_fp16_bx0_ex0_bt128x128x32_wt32x32x8_ws1x1_wr2x2_ta1x8x2x1_1x4x1x64_tb1x8x2x1_1x4x1x64_gkgs",563,9075229,16119.412078,0.8192,8172,28763,3708.184699
"SubTensorOpWithScalar1d",2774,8900046,3208.379957,0.8034,1081,15503,1785.945155
"igemm_wrw_gtcx3_nhwc_fp16_bx0_ex1_bt256x256x32_wt32x32x8_ws2x2_wr2x2_ta1x4x1x8_1x8x1x32_tb1x4x1x8_1x8x1x32_gkgs",238,7955597,33426.878151,0.7182,14782,49634,4192.493464
"igemm_wrw_gtcx3_nhwc_fp16_bx0_ex0_bt256x256x32_wt32x32x8_ws2x2_wr2x2_ta1x4x1x8_1x8x1x32_tb1x4x1x8_1x8x1x32_gkgs",360,6661235,18503.430556,0.6013,11056,51717,5898.362328
```

From the above, we can clearly identify that the most expensive kernels in this application are now collective kernels (`rccl_main_kernel`) and kernels that are relevant to this AI model (resnet): gemm and batchnorm, both forward and backwards.  Note that the rccl kernel doesn't appear as a top kernel in the single-process run.

> Note: rocprofv3 will give each output file a unique name based on process ID.  For scale out jobs, it's useful to control the output directory to a local scratch space to prevent collision.  You can control both the output file basename and directory with the arguments `-o` and `-d` respectively.

Rocprofv3 can also output the same information in a trace-based view, that can be visualized with Perfetto.  To configure this, change the output format from the default (csv) to pftrace: 

```bash 
srun --ntasks=4 rocprofv3 --stats --sys-trace --kernel-trace --output-format pftrace -- python3 <application>.py
```

Example traces look like the image below, where users can zoom in and out and learn detailed information about the performance of each kernel.

![Rocprofv3 trace of the training application](rocprofv3-trace-output.png)

In the image above, the highlighted kernel (small black box with a line coming from it) is a a rocclr_copyBuffer, and the orange boxes to the right are more collective kernels.  The green kernels to left are all compute kernels from the model's forward and backward pass.

> Rocprofv3 will serialize kernel dispatches to ensure that only one dispatch is ever in flight.  So, for some kernels that are expected to be overlapped (communication/computation, for example), rocprofv3 will not provide valuable profiling information for measuring stream concurrency.  For determining which kernels are most expensive, however, rocprofv3 can be very useful.


