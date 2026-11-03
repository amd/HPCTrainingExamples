/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*
 * rocSHMEM collective example.
 *
 * Minimal broadcast example adapted from rocSHMEM's own examples/tests.
 * PE 0's 10-element symmetric buffer is broadcast to every PE.
 */
#include <rocshmem/rocshmem.hpp>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

using namespace rocshmem;

#define HIPCHECK(cmd) do {                                                  \
    hipError_t e = cmd;                                                     \
    if (e != hipSuccess) {                                                  \
        printf("Failed: HIP error %s:%d '%s'\n",                           \
               __FILE__, __LINE__, hipGetErrorString(e));                  \
        exit(1);                                                            \
    }                                                                       \
} while (0)

/* rocSHMEM's launcher (mpirun) sets this to distinguish local ranks */
static int get_launcher_local_rank()
{
    char *s = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (s) return atoi(s);
    s = getenv("SLURM_LOCALID");
    return s ? atoi(s) : 0;
}

#define PE_ROOT 0

__global__ void init_buffers(int *source, int *dest, int nelem)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nelem) {
        source[tid] = tid;
        dest[tid] = -1;
    }
}

__global__ void broadcast_test(int *source, int *dest, size_t nelem, int root,
                               rocshmem_team_t team)
{
    __shared__ rocshmem_ctx_t ctx;
    int64_t ctx_type = 0;
    rocshmem_wg_ctx_create(ctx_type, &ctx);
    rocshmem_ctx_int_broadcast_wg(ctx, team, dest, source, nelem, root);
    rocshmem_ctx_quiet(ctx);
    __syncthreads();
    rocshmem_wg_ctx_destroy(&ctx);
}

__global__ void check_buffers(const int *dest, int nelem, int *err_count)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nelem && dest[tid] != tid) atomicAdd(err_count, 1);
}

int main(int argc, char *argv[])
{
    int count = 10;
    int errs = 0;
    int npes;
    int *d_err = nullptr;

    HIPCHECK(hipSetDevice(get_launcher_local_rank()));
    rocshmem_init();

    int my_pe = rocshmem_my_pe();
    npes = rocshmem_n_pes();
    printf("pe%d running on device %d\n", my_pe, get_launcher_local_rank());

    int *source = (int *)rocshmem_malloc(count * sizeof(int));
    int *dest = (int *)rocshmem_malloc(count * sizeof(int));
    HIPCHECK(hipMalloc(&d_err, sizeof(int)));
    if (source == NULL || dest == NULL) {
        printf("Error allocating memory from symmetric heap\n");
        rocshmem_global_exit(1);
    }

    init_buffers<<<dim3(1), dim3(256), 0, 0>>>(source, dest, count);
    HIPCHECK(hipMemset(d_err, 0, sizeof(int)));
    HIPCHECK(hipDeviceSynchronize());

    rocshmem_team_t team_bcast_world_dup = ROCSHMEM_TEAM_INVALID;
    int split_rc = rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, npes,
                                               nullptr, 0, &team_bcast_world_dup);
    if (split_rc != 0 || team_bcast_world_dup == ROCSHMEM_TEAM_INVALID) {
        printf("Error splitting team\n");
        rocshmem_global_exit(1);
    }

    rocshmem_barrier_all();
    broadcast_test<<<dim3(1), dim3(256), 0, 0>>>(
        source, dest, count, PE_ROOT, team_bcast_world_dup);
    HIPCHECK(hipDeviceSynchronize());
    rocshmem_barrier_all();

    if (my_pe != PE_ROOT) {
        check_buffers<<<dim3(1), dim3(256), 0, 0>>>(dest, count, d_err);
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipMemcpy(&errs, d_err, sizeof(int), hipMemcpyDeviceToHost));
    }

    if (errs != 0) printf("pe%d: errors=%d\n", my_pe, errs);
    if (errs == 0) {
        printf("pe%d: Run successful: Broadcast buffer has the right value\n",
               my_pe);
    }
    fflush(stdout);
    rocshmem_barrier_all();

    rocshmem_free(source);
    rocshmem_free(dest);
    HIPCHECK(hipFree(d_err));
    rocshmem_finalize();
    return 0;
}
