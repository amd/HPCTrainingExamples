# Build Instructions

```bash
module load rocm
module load omniperf/6.2.0
```
NOTE: we had to comment `!$OMP REQUIRES UNIFIED_SHARED_MEMORY` to build with `amdflang`.
```bash
amdflang -fopenmp --offload-arch=gfx942 problem.f90
```
# Test Results using Omniperf/6.2.0

## Omniperf Roofline

Running this:
```bash
export ROCPROF=rocprof
omniperf profile -n rooflines_PDF --roof-only --kernel-names -- ./a.out
```
Gives this error:
```bash
ERROR Roofline temporarily disabled in MI300
```
Running this:
```bash
export ROCPROF=rocprofv2
omniperf profile -n rooflines_PDF --roof-only --kernel-names -- ./a.out
```
Gives this error:
```bash
# ERROR Roofline temporarily disabled in MI300
```
Running this:
```bash
export ROCPROF=rocprofv3
omniperf profile -n rooflines_PDF --roof-only --kernel-names -- ./a.out
```
Gives this error:
```bash
# ERROR Incompatible profiler: rocprofv3. Supported profilers include: ['rocprofv1', 'rocprofv2', 'rocscope']
```

## Omniperf Profile
Runing this:
```bash
export ROCPROF=rocprof
omniperf profile -n v1 --no-roof -- ./a.out
```
Gives this error:
```bash
# ERROR gfx942 is not enabled in rocprofv1
```
Running this:
```bash
export ROCPROF=rocprofv2
omniperf profile -n v1 --no-roof -- ./a.out
```
Produces and output and informs that the rooflines are temporarily disabled in MI300:
```bash
# Omniperf runs but at the end it says "Roofline temporarily disabled in MI300"
```
Running this:
```bash
export ROCPROF=rocprofv3
omniperf profile -n v1 --no-roof -- ./a.out
```
Gives this error:
```bash
# ERROR Incompatible profiler: rocprofv3. Supported profilers include: ['rocprofv1', 'rocprofv2', 'rocscope']
```

## Omniperf Analyze
Running this:
```bash
export ROCPROF=rocprofv2
omniperf analyze -p workloads/v1/* -p workloads/v2/* --block 7.1.0 7.1.1 7.1.2 7.1.0: Grid size 7.1.1: Workgroup size 7.1.2: Total Wavefronts
```
Produces output. The kernel names are:
```bash
__nv_MAIN__TARGET_F1L34_1_.kd  // first kernel call at line 34 (L34)
__nv_MAIN__TARGET_F1L49_2_.kd  // second kernel call at line 49 (L49)
```

