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
 * RCCL point-to-point example.
 *
 * This is the RCCL equivalent of ../MPI-examples/pt2pt.cpp: rank 0 sends a GPU buffer to rank 1, which validates the received values.
 * MPI is only used to bootstrap the ranks and to distribute the RCCL unique id needed to create the RCCL communicator.
 */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#define HIPCHECK(cmd) do {                                                  \
    hipError_t e = cmd;                                                     \
    if (e != hipSuccess) {                                                  \
        printf("Failed: HIP error %s:%d '%s'\n",                           \
               __FILE__, __LINE__, hipGetErrorString(e));                   \
        MPI_Abort(MPI_COMM_WORLD, 1);                                       \
    }                                                                       \
} while (0)

#define NCCLCHECK(cmd) do {                                                 \
    ncclResult_t r = cmd;                                                   \
    if (r != ncclSuccess) {                                                 \
        printf("Failed: RCCL error %s:%d '%s'\n",                          \
               __FILE__, __LINE__, ncclGetErrorString(r));                  \
        MPI_Abort(MPI_COMM_WORLD, 1);                                       \
    }                                                                       \
} while (0)

int main(int argc, char *argv[])
{
    int rank, size, i;
    int *h_buf;
    int *d_buf;
    int bufsize=10;
    int deviceID=0;
    ncclUniqueId id;
    ncclComm_t comm;
    hipStream_t stream;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (size < 2)
    {
        printf("Please run with two processes.\n");fflush(stdout);
        MPI_Finalize();
        return 0;
    }
    //set device
    HIPCHECK(hipSetDevice(rank));

    //check device ID
    HIPCHECK(hipGetDevice(&deviceID));
    printf("rank%d running on device %d\n", rank, deviceID);

    //rank 0 creates the RCCL id and broadcasts it to the other ranks over MPI
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    //create the RCCL communicator and a stream to run RCCL ops on
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    HIPCHECK(hipStreamCreate(&stream));

    //allocate memory
    h_buf=(int*) malloc(sizeof(int)*bufsize);
    HIPCHECK(hipMalloc(&d_buf,bufsize*sizeof(int)));

    //initialize
    if (rank == 0) { for (i=0; i<bufsize; i++) h_buf[i] = i; }
    if (rank == 1) { for (i=0; i<bufsize; i++) h_buf[i] = -1; }
    HIPCHECK(hipMemcpy(d_buf, h_buf, (bufsize) * sizeof(int), hipMemcpyHostToDevice));

    // communication
    if (rank == 0) {
       NCCLCHECK(ncclSend(d_buf, bufsize, ncclInt, 1, comm, stream)); }

    if (rank == 1) {
       NCCLCHECK(ncclRecv(d_buf, bufsize, ncclInt, 0, comm, stream)); }

    HIPCHECK(hipStreamSynchronize(stream));

    // validate results
    if (rank == 1)
    {
        int flag=0;
        HIPCHECK(hipMemcpy(h_buf, d_buf, (bufsize) * sizeof(int), hipMemcpyDeviceToHost));
        for (i=0; i<bufsize; i++) { if (h_buf[i] != i) flag++; }
        if(flag==0) printf("Run successful: received buffer has the right value\n");
        fflush(stdout);
    }

    free(h_buf);
    HIPCHECK(hipFree(d_buf));
    HIPCHECK(hipStreamDestroy(stream));
    ncclCommDestroy(comm);
    MPI_Finalize();
    return 0;
}
