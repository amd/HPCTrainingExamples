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
 * RCCL collective example.
 *
 * This is the RCCL equivalent of ../MPI-examples/collective.cpp.
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

/* A simple test of GPU-to-GPU Reduce over RCCL */
int main( int argc, char *argv[] )
{
    int errs = 0;
    int rank, size, root;
    int *d_sendbuf, *d_recvbuf;
    int *h_buffer,i;
    int count;
    int deviceID=0;
    ncclUniqueId id;
    ncclComm_t comm;
    hipStream_t stream;

    MPI_Init( &argc, &argv );

    /* Determine the sender and receiver */
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    count=10;
    root=0;

    //set device
    HIPCHECK(hipSetDevice(rank%8));

    //check device ID
    HIPCHECK(hipGetDevice(&deviceID));
    printf("rank%d running on device %d\n", rank, deviceID);

    //rank 0 creates the RCCL id and broadcasts it to the other ranks over MPI
    if (rank == 0) NCCLCHECK(ncclGetUniqueId(&id));
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    //create the RCCL communicator and a stream to run RCCL ops on
    NCCLCHECK(ncclCommInitRank(&comm, size, id, rank));
    HIPCHECK(hipStreamCreate(&stream));

    //allocate memory on host
    h_buffer = (int *)malloc( count * sizeof(int) );

    //allocate memory on device
    HIPCHECK(hipMalloc(&d_sendbuf,count*sizeof(int)));
    HIPCHECK(hipMalloc(&d_recvbuf,count*sizeof(int)));

    //initialize send and receive buffers
    for (i=0; i<count; i++) h_buffer[i] = i;
    HIPCHECK(hipMemcpy(d_sendbuf, h_buffer, (count) * sizeof(int), hipMemcpyHostToDevice));

    HIPCHECK(hipMemset(d_recvbuf,0,count*sizeof(int)));

    //GPU-to-GPU Reduce over RCCL
    NCCLCHECK(ncclReduce(d_sendbuf, d_recvbuf, count, ncclInt, ncclSum, root, comm, stream));

    HIPCHECK(hipStreamSynchronize(stream));

    //validate results
    if (rank == root) {
       for (i=0; i<count; i++) h_buffer[i] = 0;
       HIPCHECK(hipMemcpy(h_buffer, d_recvbuf, (count) * sizeof(int), hipMemcpyDeviceToHost));
       for (i=0; i<count; i++) {
          if (h_buffer[i] != i * size) {
              errs++;
           }
        }
       if(errs!=0) printf("errors=%d\n", errs);
       if(errs==0) printf("Run successful: Reduced buffer has the right value\n");
    }

    HIPCHECK(hipFree(d_sendbuf));
    HIPCHECK(hipFree(d_recvbuf));
    free( h_buffer );
    HIPCHECK(hipStreamDestroy(stream));
    ncclCommDestroy(comm);

    MPI_Finalize();
    return 0;
}
