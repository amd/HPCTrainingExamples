# Ghost Exchange Example: HIP Implementation with GPU kernel / GPU MPI exchange overlap

In this example, the kernel computations advancing the solution happen on the GPU and overlap with the MPI exchanges, which also happen on the GPU.
As in [Ver6](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver6), the solution array and the communication buffers are allocated on the GPU with `hipMalloc` and GPU aware MPI is leveraged, therefore one could `unset HSA_XNACK`.
This example can be seen as a variation of [Ver8](https://github.com/amd/HPCTrainingExamples/tree/main/MPI-examples/GhostExchange/GhostExchange_ArrayAssign_HIP/Ver8), where the MPI exchanges have been moved from the CPU to the GPU.

## SDMA engines vs blit kernels

Within a node, GPU aware MPI libraries that copy through the HSA runtime (e.g. Open MPI with UCX) perform the GPU to GPU copies either with the SDMA copy engines (default) or with blit kernels running on the compute units (`export HSA_ENABLE_SDMA=0`): only the first can overlap with a kernel that is using all the compute units.
The solution array is stored row by row, so the up and down halos are rows, contiguous in memory, and MPI sends them directly from the array. The left and right halos are columns, whose values are `jstride` doubles apart: before sending, a GPU kernel packs them into a contiguous buffer, and after receiving, another kernel unpacks the buffer into the ghost columns. With a `-x 1` decomposition there are no left and right neighbors, so only the up and down exchanges happen and no packing is needed, which is the simplest case to observe the overlap. Note that `-y 1` is not equivalent: it leaves only the left and right exchanges, and all of them need packing and unpacking.

Build the example

```
cd Ver9
mkdir build && cd build
cmake ..
make -j
```

and run it once with the SDMA engines and once with blit kernels. The halo is 128 cells wide (`-h 128`) to make the communication time comparable to the computation time: each up and down message is 128 rows of 20256 doubles (about 21 MB), and the exchange takes almost as long as the inner cells kernel. With a thin halo the exchange is so short that hiding it barely changes the total time.

```
unset HSA_XNACK HSA_ENABLE_SDMA
mpirun -n 4 --bind-to core --map-by ppr:1:numa ../../set_gpu_device_mi300a.sh ./GhostExchange -x 1 -y 4 -i 20000 -j 20000 -h 128 -t -c -I 1000
export HSA_ENABLE_SDMA=0
mpirun -n 4 --bind-to core --map-by ppr:1:numa -x HSA_ENABLE_SDMA ../../set_gpu_device_mi300a.sh ./GhostExchange -x 1 -y 4 -i 20000 -j 20000 -h 128 -t -c -I 1000
```

To check that the overlap happened, compare the `Total` times of the two runs. On one MI300A node with Open MPI and UCX, the two runs above print (median of 3 runs each):

| | `Ghost Cell Update` (s) | `Total` (s) |
| --- | ---: | ---: |
| SDMA engines (default) | 0.83 | 2.01 |
| blit kernels (`HSA_ENABLE_SDMA=0`) | 1.23 | 2.39 |

With the SDMA engines the exchange, about 0.8 ms per iteration, runs while `blur_inner` computes, and is hidden behind it. With blit kernels the copies wait for `blur_inner` to free the compute units and only then run, so `Ghost Cell Update` grows to one `blur_inner` plus the copies per iteration, and the total time grows by about 0.38 s: this is the communication time that the SDMA engines hide. The duration of `blur_inner` is reported by rocprofv3 as `AverageNs`, about 0.9 ms per call:

```
unset HSA_ENABLE_SDMA
mpirun -n 4 --bind-to core --map-by ppr:1:numa bash -c 'ROCR_VISIBLE_DEVICES=$OMPI_COMM_WORLD_LOCAL_RANK rocprofv3 --kernel-trace --stats \
    -f csv -o trace -d prof/rank$OMPI_COMM_WORLD_LOCAL_RANK -- ./GhostExchange -x 1 -y 4 -i 20000 -j 20000 -h 128 -t -c -I 100'
grep -h -e Name -e blur_inner prof/rank*/trace_kernel_stats.csv
```
