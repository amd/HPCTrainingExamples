# OpenMP complex compute constructs in C

README.md from `HPCTrainingExamples/Pragma_Examples/OpenMP/C/ComplexComputeConstructs` in the Training Examples repository

These exercises explore more complex compute constructs. We begin with breaking apart the meanings of each of the
clauses in the single combined compute directive.

First retrieve the examples for this part.

```
git clone https://github.com/amd/HPCTrainingExamples

```

## Full combined compute directive

We'll start with a baseline from the full combined compute directive

```
#pragma omp target teams distribute parallel for simd
```

Setting up the environment

```
module load amdclang
export HSA_XNACK=1
export LIBOMPTARGET_KERNEL_TRACE=1
```

This example is in the previous exercises on simple single line compute constructs

```
cd HPCTrainingExamples/Pragma_Examples/OpenMP/C/SingleLineConstructs
```

```
make saxpy_gpu_parallelfor
./saxpy_gpu_parallelfor
```

Check the output. It should be something like the following, but with some variation depending
on the GPU model you are on.

```
DEVID:  0 SGN:5 ConstWGSize:256  args: 5 teamsXthrds:( 416X 256) reqd:(   0X   0) lds_usage:0B sgpr_count:24 vgpr_count:8 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:10000000 rpc:0 n:__omp_offloading_34_8975356_saxpy_l8
Time of kernel: 0.082906
```

There are 416 teams (workgroups) of size 256. There is a low vector register usage a 8. We'll also look at the run-time of 0.082906 for comparison.


## Target directive

We'll start with what happens with just the target directive

cd HPCTrainingExamples/Pragma_Examples/OpenMP/C/ComplexComputeConstructs

Setting up the environment

```
module load amdclang
export HSA_XNACK=1
export LIBOMPTARGET_KERNEL_INFO=1
```

```
make saxpy_gpu_target
./saxpy_gpu_target
```

The output will be similar to the following:

```
DEVID:  0 SGN:3 ConstWGSize:257  args: 5 teamsXthrds:(   1X 256) reqd:(   0X   0) lds_usage:16B sgpr_count:16 vgpr_count:3 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:0 rpc:0 n:__omp_offloading_34_5c4ed40a_saxpy_l8
Time of kernel: 5.407085
```

We only have one team of 256 workgroup size. Basically we are running serial -- one thread on one team (workgroup). The runtime reflects that with 65 times longer than the combined directive.

## Teams clause

The teams exercise will add the teams clause after the target directive.

```
make saxpy_gpu_target_teams
./saxpy_gpu_target_teams
```

The output

```
DEVID:  0 SGN:3 ConstWGSize:257  args: 5 teamsXthrds:( 624X 256) reqd:(   0X   0) lds_usage:16B sgpr_count:12 vgpr_count:3 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:0 rpc:0 n:__omp_offloading_34_5c4ed40b_saxpy_l8
Time of kernel: 11.166301
```

There are 624 workgroups, but each one is doing all the work. This duplicates the effort and ends up taking twice the time as the target directive alone.
Note that this is also creating a race condition when threads are trying to write to the same location, which produces an incorrect output that is also non deterministic. One could add `num_teams(1)` to the pragma directive to require the creation of a single team, in which case no race condition can occur.

### Distribute clause

Adding the distribute clause starts to get some parallelism by partitioning the work across the workgroups. But still with only one thread per workgroup.

```
make saxpy_gpu_target_teams_distribute
./saxpy_gpu_target_teams_distribute
```

Output

```
DEVID:  0 SGN:3 ConstWGSize:257  args: 5 teamsXthrds:( 624X 256) reqd:(   0X   0) lds_usage:16B sgpr_count:24 vgpr_count:3 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:10000000 rpc:0 n:__omp_offloading_34_5c4ed40c_saxpy_l8
Time of kernel: 0.149113
```

We have more workgroups at 624 than the baseline case, but we are not using all the threads. This is using more of the compute capacity at 624/416 times as many workgroups and associated compute units. The runtime is much closer to the baseline. As a further exploration, try changing the array size in the example or trying a different kernel with more work.

### parallel for without the teams distribute clauses

As a further experiment, let's try just adding parallel for to engage all the threads on one workgroup. The directive is the following:

```
#pragma omp target parallel for
```

Building and running it

```
make saxpy_gpu_parallel_for
./saxpy_gpu_parallel_for
```

Output should be something like

```
DEVID:  0 SGN:2 ConstWGSize:256  args: 5 teamsXthrds:(   1X 256) reqd:(   0X   0) lds_usage:32B sgpr_count:25 vgpr_count:17 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:0 rpc:0 n:__omp_offloading_34_5c4ed416_saxpy_l8
Time of kernel: 0.126748
```

This gives a pretty good runtime while using fewer GPU compute units.

## Split multi-level directive

Build both the collapse and split level C examples.

```
make saxpy_gpu_collapse
./saxpy_gpu_collapse
make saxpy_gpu_split_level
./saxpy_gpu_split_level
```

Compare the output from LIBOMPTARGET_KERNEL_TRACE=1.

```
DEVID:  0 SGN:5 ConstWGSize:256  args: 6 teamsXthrds:(3907X 256) reqd:(   0X   0) lds_usage:0B sgpr_count:29 vgpr_count:17 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:1000000 rpc:0 md:0 md_LB:-1 md_UB:-1 Max Occupancy: 8 Achieved Occupancy: 100% n:__omp_offloading_34_5c4ed40e_saxpy_l9

Time of kernel: 0.027777
```

```
DEVID:  0 SGN:3 ConstWGSize:257  args: 6 teamsXthrds:( 416X 256) reqd:(   0X   0) lds_usage:36B sgpr_count:27 vgpr_count:24 sgpr_spill_count:0 vgpr_spill_count:0 tripcount:1000 rpc:0 md:0 md_LB:-1 md_UB:-1 Max Occupancy: 8 Achieved Occupancy: 50% n:__omp_offloading_34_5c4ed411_saxpy_l9

Time of kernel: 0.027449
```

On your own: try different array sizes and ratios of iterations between the loop levels.

