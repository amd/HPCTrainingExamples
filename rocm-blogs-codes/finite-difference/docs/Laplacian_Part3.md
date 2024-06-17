<!---
Copyright (c) 2023 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
--->
# Finite difference method - Laplacian part 3

In the previous two Laplacian posts, we developed a HIP implementation of a finite-difference code designed around a Laplace operator
and applied two possible code optimizations to optimize memory movement between the L2 cache and global memory. 
This third part will cover some additional optimizations and general tips to fine tune the performance of the kernel. 
To quickly review, recall
that the Laplacian takes the form of a divergence of a gradient of a scalar field $u(x,y,z)$:

$$\nabla \cdot \nabla u = \nabla^2 u = \frac{\partial^2u}{\partial x^2} + \frac{\partial^2u}{\partial y^2} + \frac{\partial^2u}{\partial z^2},$$

The performance of the baseline HIP implementation we started off with in [Part 1](./Laplacian_Part1.md#hip-implementation) achieved around 50%[^1] of the theoretical peak. However, based on initial `rocprof` analyses, 
we projected that the finite difference kernel should reach at least 71%[^1] of the peak. To meet this goal, we applied two optimizations:

1. Introduced loop tiling to explicitly reuse loaded stencils
2. Reordered the read access pattern of the stencil points

Please see the reordering read access patterns section in [Part 2](./Laplacian_Part2.md#reorder-read-access-patterns) for the full code implementation.
With these changes, we have reached 95% of the performance projection, but we still have some open questions:
- The previous optimizations required manual tuning of certain parameters that could lead to strange performance 
characteristics where performance suddenly drops from a large tiling factor. Could fixing the root cause of this performance drop get us closer to the target figure of merit (FOM) 
as defined in [Part 1](./Laplacian_Part1.md#performance)?
- We have focused solely on optimizing the kernel's cache and data reuse of the fetch operations. Could we make some gains by improving the same aspects of the write operations?
- The optimizations we have introduced required non-trivial code changes. Are there alternative optimizations we could leverage 
to gain significant performance without the increase in code complexity?

This blog post will explore some of these remaining questions. The next few sections will introduce and discuss the following concepts:

1. Generating temporary files to understand register usage and scratch memory
1. Applying launch bounds to control register usage
1. Applying nontemporal stores to free up more cache

## Register pressure and scratch use

In the loop tiling optimization described in the previous post, a tiling factor `m=16` caused the kernels'
FOM to deteriorate. The `rocprof` metric `FETCH_SIZE` rose to over 4x the theoretical limit and the `L2CacheHit`
metric dropped below 50%. We suspect the high tiling factor proliferated the register usage, causing spillage.
To this end, we introduce a new compilation flag `--save-temps` which tells the compiler to generate important
information regarding register usage, spills, occupancy, and much more for every GPU kernel. It also includes 
instruction set architecture (ISA) dumps for both host and device code. A future lab notes post will cover the 
AMD GPU ISA in full detail.

We examine four key metrics:

1. SGPR
2. VGPR
3. Scratch
4. Occupancy

SGPR and VGPR refer to the scalar and vector registers, Scratch refers to scratch memory which could be an
indicator of register spills, and Occupancy represents the maximum number of wavefronts that can run on the 
Execution Unit (EU). Note that statistics for register and scratch use can be found directly from `rocprof` output files
whereas other details like occupancy and ISA assembly can only be found from the temporary files.
Users can simply uncomment the line `#TEMPS=true` in the provided 
`makefile` to generate temporary files located in a file named `temps/laplacian_dp_kernel1-hip.s` 
containing this information. Here is some sample output for the baseline HIP kernel 1:

```
  .section  .AMDGPU.csdata
; Kernel info:
; codeLenInByte = 520
; NumSgprs: 18
; NumVgprs: 24
; NumAgprs: 0
; TotalNumVgprs: 24
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 2
; NumSGPRsForWavesPerEU: 18
; NumVGPRsForWavesPerEU: 24
; AccumOffset: 24
; Occupancy: 8
; WaveLimiterHint : 1
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 8
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 2
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 5
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
```

In addition to the ISA dump, there is a lot of information to unpack here. 
We defer all interested readers to
[this presentation](https://www.olcf.ornl.gov/wp-content/uploads/Intro_Register_pressure_ORNL_20220812_2083.pdf)
as well as the [register pressure](../../register-pressure/README.md) post for more details 
on registers, scratch, occupancy, and more.

The table below contains the above four key metrics for the baseline and optimized
kernel with various tiling factors `m`:

|                                  | SGPR | VGPR | Scratch | Occupancy |
|----------------------------------|:----:|:----:|:-------:|:---------:|
| Kernel 1 - Baseline              |   18 |   24 |     0   |      8    |
| Kernel 3 - Reordered loads m=1   |   24 |   18 |     0   |      8    |
| Kernel 3 - Reordered loads m=2   |   26 |   28 |     0   |      8    |
| Kernel 3 - Reordered loads m=4   |   34 |   54 |     0   |      8    |
| Kernel 3 - Reordered loads m=8   |   52 |   90 |     0   |      5    |
| Kernel 3 - Reordered loads m=16  |   90 |  128 |   180   |      4    |

There is a strong correlation between the tiling factor and register/scratch usage. At `m=16`, 
register usage has increased to the point where it "spills" into scratch space, that is, registers 
that no longer fit in the register space are offloaded into global memory. There is also a strong
but inverse correlation between occupancy, defined as waves per EU, and register use - as register usage goes up, 
the occupancy goes down. So what can be done to prevent spills or increase occupancy?

## Launch bounds

One quick way to control register usage is to apply launch bounds to the kernel. By default, the 
HIP compiler restricts the number of registers per thread based on the maximum allowable 
thread block size of 1024 threads. If the thread block size is known at compile time, it is a good 
practice to set the launch bounds for the kernel. Setting launch bounds takes the following 
arguments:

```c++
__launch_bounds__(MAX_THREADS_PER_BLOCK,MIN_WAVES_PER_EU)
```

The first argument `MAX_THREADS_PER_BLOCK` informs the compiler about the thread block dimensions 
such that it can optimize the register usage for the particular block size. The second argument
`MIN_WAVES_PER_EU` is an optional argument that specifies the minimum number of wavefronts required
to be active on each EU. By default, this second value is set to 1 and does not need to be modified whereas 
the default `MAX_THREADS_PER_BLOCK` value of 1024 needs to be changed because we are not using all
1024 threads.

So far, we have been using a thread block size of 256 $\times$ 1 $\times$ 1, so here is how to set 
the launch bounds with `MAX_THREADS_PER_BLOCK = 256` for Kernel 3:

```c++
template <typename T>
__launch_bounds__(256)
__global__ void laplacian_kernel(...) { 

...
```

Let us designate this one-line change "Kernel 4" and examine the impact on register and scratch space usage:

|                                           | SGPR | VGPR | Scratch | Occupancy |
|-------------------------------------------|:----:|:----:|:-------:|:---------:|
| Kernel 1 - Baseline                       |   18 |   24 |     0   |      8    |
| Kernel 3/Kernel 4 - Reordered loads m=1   |   24/24 |   18/18 |     0/0 |      8/8  |
| Kernel 3/Kernel 4 - Reordered loads m=2   |   26/26 |   28/28 |     0/0 |      8/8  |
| Kernel 3/Kernel 4 - Reordered loads m=4   |   34/34 |   54/54 |     0/0 |      8/8  |
| Kernel 3/Kernel 4 - Reordered loads m=8   |   52/52 |   90/94 |     0/0  |     5/5  |
| Kernel 3/Kernel 4 - Reordered loads m=16  |   90/84 |  128/170 |   180/0   |   4/2  |

Applying launch bounds to tiling factors `m=4` and below had no effect on the register usage.
When `m=8`, only the VGPR increased slightly, whereas for `m=16` the VGPR increased greatly and
the scratch usage was eliminated altogether. Notice that the occupancy drops significantly raising questions
as to whether this will negatively affect performance at `m=16`. Let us look at the FOM performance:

|                                  | Speedup |  % of target |
|----------------------------------|:-------:|:------------:|
| Kernel 1 - Baseline              |   1.00  |      69.4%   |
| Kernel 3 - Reordered loads m=1   |   1.20  |      82.9%   |
| Kernel 3 - Reordered loads m=2   |   1.28  |      88.9%   |
| Kernel 3 - Reordered loads m=4   |   1.34  |      93.1%   |
| Kernel 3 - Reordered loads m=8   |   1.37  |      94.8%   |
| Kernel 3 - Reordered loads m=16  |   0.42  |      29.4%   |
| Kernel 4 - Launch bounds m=1     |   1.20  |      82.9%   |
| Kernel 4 - Launch bounds m=2     |   1.28  |      88.9%   |
| Kernel 4 - Launch bounds m=4     |   1.34  |      93.1%   |
| Kernel 4 - Launch bounds m=8     |   1.39  |      96.1%   |
| Kernel 4 - Launch bounds m=16    |   1.34  |      93.2%   |
    
Unsurprisingly, kernels where the launch bounds had no effect on register, scratch, or occupancy
had the same performance as before. There is a clear correlation between how much the launch bounds 
affected the SGPR, VGPR, Scratch, and Occupancy statistics and how much performance was gained. 
Kernels with tiling factors `m=8` and `m=16` saw improvements in their respective performances. 
Let us examine the corresponding `rocprof` metrics:

|                                  | FETCH_SIZE (GB) | Fetch efficiency (%) | L2CacheHit (%) |
|----------------------------------|:---------------:|:--------------------:|:--------------:|
| Theoretical                      |           1.074 |                   -  |    -  |
| Kernel 1 - Baseline              |           2.014 |                 53.3 |  65.0 |
| Kernel 3 - Reordered loads m=1   |           1.347 |                 79.7 |  72.0 |
| Kernel 3 - Reordered loads m=2   |           1.166 |                 92.1 |  70.6 |
| Kernel 3 - Reordered loads m=4   |           1.107 |                 97.0 |  68.8 |
| Kernel 3 - Reordered loads m=8   |           1.080 |                 99.4 |  67.7 |
| Kernel 3 - Reordered loads m=16  |           3.915 |                 27.4 |  44.5 |
| Kernel 4 - Launch bounds m=1     |           1.346 |                 79.8 |  72.0 |
| Kernel 4 - Launch bounds m=2     |           1.167 |                 92.1 |  70.6 |
| Kernel 4 - Launch bounds m=4     |           1.107 |                 97.0 |  68.8 |
| Kernel 4 - Launch bounds m=8     |           1.080 |                 99.4 |  67.3 |
| Kernel 4 - Launch bounds m=16    |           1.094 |                 98.2 |  66.1 |

> **NOTE**: Although neither `WRITE_SIZE` nor Write efficiency(%) were shown for these experiments, the reported `WRITE_SIZE` and Write efficiency(%) for `Kernel 3 - Reordered loads m=16` was 2.547 GB and 41.7%, respectively. Kernels without scratch spills have nearly 100% write efficiency.
 
Now that `m=16` with launch bounds no longer spills registers into scratch, we saw significant 
gains in the speedup, fetch efficiency, and `L2CacheHit`. Users working on optimizing kernels 
with high register usage can quickly gain performance back by applying launch bounds.
Although the kernel with `m=8` has much lower register usage compared to `m=16`, applying launch bounds 
still had an impact on the VGPR usage thereby increasing overall performance just enough to become the 
best performing kernel yet. The new FOM from the kernel with `m=8` is still a little 
short of our projected target, so let us explore another code optimization.
    
## Nontemporal memory access

Most of our optimization efforts focused on improving spatial locality, however what we 
have not considered yet is temporal locality - that is to say, 
how to prioritize the caching of variables in time. Both loading the elements of `u` 
as well as storing the elements of `f` will occupy cache lines. The difference though 
is that based on the data layout described in [Part 1](./Laplacian_Part1.md#data-layout), each element of `u` could theoretically be reused up to six times whereas each 
element of `f` is accessed only once. Thus, we can use the clang's builtin nontemporal 
store intrinsic to let `f` bypass the L2 cache, thereby increasing the available cache for 
entries in `u`. It should also be noted that these intrinsics are specific to AMD GPUs.
    
The AMD clang compiler provides two overloaded builtins allowing generation of non-temporal 
loads and stores:

```c++
T __builtin_nontemporal_load(T *addr);
void __builtin_nontemporal_store(T value, T *addr);
```

In the Laplacian examples, we only need the nontemporal stores. Let us first apply this builtin to the initial
baseline kernel:

<table class="scrollable-table">
<tr>
<th>
Kernel 1 (Before)
</th>
<th>
Kernel 1 (After)
</th>
</tr>
<tr>
<td>
    
```c++
f[pos] = u[pos] * invhxyz2
       + (u[pos - 1]     + u[pos + 1]) * invhx2
       + (u[pos - nx]    + u[pos + nx]) * invhy2
       + (u[pos - slice] + u[pos + slice]) * invhz2;
```

</td>
<td>

```c++
__builtin_nontemporal_store(u[pos] * invhxyz2
       + (u[pos - 1]     + u[pos + 1]) * invhx2
       + (u[pos - nx]    + u[pos + nx]) * invhy2
       + (u[pos - slice] + u[pos + slice]) * invhz2, &f[pos]);
```

</td>
</tr>
</table>

To assess the impact of this simple code modification, we compare the performance of the baseline implementation, with and without
the nontemporal store, with that of Kernel 3 when `m=1`:

|                                  | Speedup |  % of target |
|----------------------------------|:-------:|:------------:|
| Kernel 1 - Baseline              |   1.00  |      69.4%   |
| Kernel 1 - Nontemporal store     |   1.19  |      82.5%   |
| Kernel 3 - Reordered loads m=1   |   1.20  |      82.9%   |

The improvement from changes in a single line of code are comparable to that of refactoring the entire baseline kernel to leverage
loop tiling and reordered memory access patterns. Examining the `rocprof` stats:

|                                  | FETCH_SIZE (GB) | Fetch efficiency (%) | L2CacheHit (%) |
|----------------------------------|:---------------:|:--------------------:|:--------------:|
| Theoretical                      |           1.074 |                   -  |    -  |
| Kernel 1 - Baseline              |           2.014 |                 53.3 |  65.0 |
| Kernel 1 - Nontemporal store     |           1.429 |                 75.2 |  71.4 |
| Kernel 3 - Reordered loads m=1   |           1.347 |                 79.7 |  72.0 |

The statistics are also comparable between the nontemporal store and reordered loads when `m=1`. These findings suggest that
leveraging the overloaded builtins for nontemoral memory access could actually be the first optimization users apply, as it is a 
"lower hanging fruit" requiring modifications to only a single line of code. The obvious next question is what happens
when you combine the builtin nontemporal store with that of loop tiling factor `m=8` and launch bounds? 
Let us again perform a one-line modification to kernel 4:

<table>
<tr>
<th>
Kernel 4 (Before)
</th>
<th>
Kernel 5 (After)
</th>
</tr>
<tr>
<td>
    
```c++
f[pos + n*nx] = Lu[n];
```

</td>
<td>

```c++
__builtin_nontemporal_store(Lu[n],&f[pos + n*nx]);
```

</td>
</tr>
</table>

This new Kernel 5 is an accumulation of the optimizations involving loop tiling, reordering loads, applying launch bounds, and leveraging
nontemporal stores. The performance when the looping tiling factor `m=8` can be seen below:
    
|                                  | Speedup |  % of target |
|----------------------------------|:-------:|:------------:|
| Kernel 1 - Baseline              |   1.00  |      69.4%   |
| Kernel 3 - Reordered loads m=8   |   1.37  |      94.8%   |
| Kernel 4 - Launch bounds m=8     |   1.39  |      96.1%   |
| Kernel 5 - Nontemporal store m=8 |   1.44  |       100%   |
    
With all the optimizations combined, this enabled us to achieve a 1.44x speedup and meet 100% of our target!  
Let us examine the `rocprof` metrics once again:

|                                  | FETCH_SIZE (GB) | Fetch efficiency (%) | L2CacheHit (%) |
|----------------------------------|:---------------:|:--------------------:|:--------------:|
| Theoretical                      |           1.074 |                   -  |    -  |
| Kernel 1 - Baseline              |           2.014 |                 53.3 |  65.0 |
| Kernel 3 - Reordered loads m=8   |           1.080 |                 99.4 |  67.7 |
| Kernel 4 - Launch bounds m=8     |           1.080 |                 99.4 |  67.3 |
| Kernel 5 - Nontemporal store m=8 |           1.074 |                  100 |  67.4 |

Over the course of these improvements, we have reduced the number of loads from global memory by half.
The measured fetch and write sizes have reached the theoretical limits so further performance improvements must 
come from other areas. Since the reported effective memory bandwidth has reached the projected goal, there is likely
little room left for further improvement. 

## Summary and next steps

The third part of the Laplacian finite difference series introduces two additional optimizations, both requiring changes involving 
only a single line of source code. We also introduced readers to temporary files that give further insight into
a kernel's scratch usage, register pressure, and occupancy, all of which can be simply altered by applying launch bounds to the kernel. 
When compared to leveraging loop tiling and reordered memory loads, applying built-in non-temporal loads is easier to implement and 
provides a significant performance boost to the initial HIP implementation, therefore it should be preferred to kernel refactoring. 
We reiterate though, that these built-in intrinsics are not portable and are specific to AMD GPUs.
All four optimizations presented thus far, when combined, enables our effective memory bandwidth to reach its projected goal.

However, there are still some outstanding questions. The last three posts of this Laplacian series focused on optimizations tailored towards 
a problem size of `nx,ny,nz = 512, 512, 512` run on a single GCD of the MI250X GPU. What happens if we run kernel 5 on other hardware and
problem sizes? Will other performance issues arise? The next and final post in this Laplacian series will explore these 
in depth.

[Accompanying code examples](https://github.com/amd/amd-lab-notes/tree/release/finite-difference/examples)

If you have any questions or comments, please reach out to us on GitHub [Discussions](https://github.com/amd/amd-lab-notes/discussions)

[^1]:Testing conducted using ROCm version 5.3.0-63 and MI250X single GCD. Benchmark results are not validated performance numbers, and are provided only to demonstrate relative performance improvements of code modifications. Actual performance results depend on multiple factors including system configuration and environment settings, reproducibility of the results is not guaranteed.
