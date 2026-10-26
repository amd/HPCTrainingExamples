# OpenMP workdistribute Example: How to port workshare to GPU

This example demonstrates the OpenMP 6.0 `target teams workdistribute`
construct for Fortran whole-array assignments.
The subdirectory cpu_code contains the CPU OpenMP code with workshare to parallelize on the CPU. The GPU equivalent with workdistribute is in the gpu_code folder.

Allocate a compute node with a GPU, then load ROCm 10.1:

```bash
salloc -N 1 --gpus=1 -p PPAC_MI300A_CPX
module load rocm/10.1.0
export FC=amdflang
make 
./<exe> #depending on CPU or GPU code has a different name
```
