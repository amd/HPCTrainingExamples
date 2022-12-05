/*
Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
 
/* A simple test of GPU-Aware Reduce */
int main( int argc, char *argv[] )
{
    int errs = 0;
    int rank, size, root;
    int *d_sendbuf, *d_recvbuf;
    int *h_buffer,i;
    int minsize = 2, count;
    int deviceID=0;
    MPI_Comm comm;
 
    MPI_Init( &argc, &argv );
 
    comm = MPI_COMM_WORLD;
    /* Determine the sender and receiver */
    MPI_Comm_rank( comm, &rank );
    MPI_Comm_size( comm, &size );
    count=10; 
    root=0;

    //set device
    hipSetDevice(rank%8);

    //check device ID
    hipGetDevice(&deviceID);
    printf("rank%d running on device %d\n", rank, deviceID);

    //allocate memory on host
    h_buffer = (int *)malloc( count * sizeof(int) );

    //allocate memory on device
    hipMalloc(&d_sendbuf,count*sizeof(int));
    hipMalloc(&d_recvbuf,count*sizeof(int));

    //initialize send and receive buffers
    for (i=0; i<count; i++) h_buffer[i] = i;
    hipMemcpy(d_sendbuf, h_buffer, (count) * sizeof(int), hipMemcpyHostToDevice);

    hipMemset(d_recvbuf,0,count*sizeof(int));

    //GPU-Aware Reduce    
    MPI_Reduce( d_sendbuf, d_recvbuf, count, MPI_INT, MPI_SUM, root, comm );


    //validate results
    if (rank == root) {
       for (i=0; i<count; i++) h_buffer[i] = 0;
       hipMemcpy(h_buffer, d_recvbuf, (count) * sizeof(int), hipMemcpyDeviceToHost);
       for (i=0; i<count; i++) {
          if (h_buffer[i] != i * size) {
              errs++;
           }
        }
       if(errs!=0) printf("errors=%d\n", errs);
       if(errs==0) printf("Run successful: Reduced buffer has the right value\n");
    }

    hipFree(d_sendbuf);
    hipFree(d_recvbuf);
    free( h_buffer );

    MPI_Finalize();
    return 0;
}

