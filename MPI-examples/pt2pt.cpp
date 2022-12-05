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

int main(int argc, char *argv[])
{
    int rank, size, i;
    int *h_buf;
    int *d_buf;
    int bufsize=10;
    int deviceID=0;
    MPI_Status status;
 
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
    hipSetDevice(rank);

    //check device ID
    hipGetDevice(&deviceID);
    printf("rank%d running on device %d\n", rank, deviceID);

    //allocate memory
    h_buf=(int*) malloc(sizeof(int)*bufsize);
    hipMalloc(&d_buf,bufsize*sizeof(int));

    //initialize
    if (rank == 0)
    {
        for (i=0; i<bufsize; i++)
            h_buf[i] = i;
	hipMemcpy(d_buf, h_buf, (bufsize) * sizeof(int), hipMemcpyHostToDevice);
    }

    if (rank == 1)
    {
        for (i=0; i<bufsize; i++)
            h_buf[i] = -1;
	hipMemcpy(d_buf, h_buf, (bufsize) * sizeof(int), hipMemcpyHostToDevice);
    }

    // communication
    if (rank == 0) {
       MPI_Send(d_buf, bufsize, MPI_INT, 1, 123, MPI_COMM_WORLD); }

    if (rank == 1) {
       MPI_Recv(d_buf, bufsize, MPI_INT, 0, 123, MPI_COMM_WORLD, &status); }

    // validate results
    if (rank == 1)
    {
	int flag=0;
        hipMemcpy(h_buf, d_buf, (bufsize) * sizeof(int), hipMemcpyDeviceToHost);	
        for (i=0; i<bufsize; i++)
        {
            if (h_buf[i] != i){
		flag++;
                printf("Error: buffer[%d] = %d but is expected to be %d\n", i, h_buf[i], i);
	    }
        }
	if(flag==0) printf("Run successful: received buffer has the right value\n");
        fflush(stdout);
    }

    free(h_buf);
    hipFree(d_buf);
    MPI_Finalize();
    return 0;
}
 
