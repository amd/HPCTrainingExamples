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
 * rocSHMEM one-sided point-to-point example.
 *
 * This is the rocSHMEM equivalent of ../MPI-examples/pt2pt.cpp:
 * PE 0 sends a 10-element GPU buffer to PE 1, which validates the
 * received values.
 *
 * Unlike MPI_Send/MPI_Recv, rocSHMEM is one-sided: only PE 0 calls
 * rocshmem_putmem to write directly into PE 1's symmetric buffer.
 * rocSHMEM APIs are called from device code, so the put is issued
 * from a small kernel.
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

__global__ void put_to_peer(int *dst, int *src, size_t nelem, int peer)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid == 0) {
        rocshmem_putmem(dst, src, nelem * sizeof(int), peer);
        rocshmem_quiet();
    }
    __syncthreads();
}

int main(int argc, char *argv[])
{
    int bufsize = 10;
    int i;

    rocshmem_init();

    int my_pe = rocshmem_my_pe();
    int npes = rocshmem_n_pes();

    if (npes < 2) {
        printf("Please run with two PEs.\n");
        fflush(stdout);
        rocshmem_finalize();
        return 0;
    }

    HIPCHECK(hipSetDevice(get_launcher_local_rank()));
    printf("pe%d running on device %d\n", my_pe, get_launcher_local_rank());

    int *d_buf = (int *)rocshmem_malloc(bufsize * sizeof(int));
    if (d_buf == NULL) {
        printf("Error allocating memory from symmetric heap\n");
        rocshmem_global_exit(1);
    }

    if (my_pe == 0) {
        for (i = 0; i < bufsize; i++) d_buf[i] = i;
    }
    if (my_pe == 1) {
        for (i = 0; i < bufsize; i++) d_buf[i] = -1;
    }
    HIPCHECK(hipDeviceSynchronize());

    if (my_pe == 0) {
        put_to_peer<<<dim3(1), dim3(32), 0, 0>>>(d_buf, d_buf, bufsize, 1);
        // Kernel launch is async: ensure PE0's device-side put/quiet finished
        // before entering the cross-PE barrier.
        HIPCHECK(hipDeviceSynchronize());
    }
    rocshmem_barrier_all();

    if (my_pe == 1) {
        int flag = 0;
        for (i = 0; i < bufsize; i++) {
            if (d_buf[i] != i) {
                flag++;
                printf("Error: buffer[%d] = %d but is expected to be %d\n",
                       i, d_buf[i], i);
            }
        }
        if (flag == 0) {
            printf("Run successful: received buffer has the right value\n");
        }
        fflush(stdout);
    }

    rocshmem_free(d_buf);
    rocshmem_finalize();
    return 0;
}
