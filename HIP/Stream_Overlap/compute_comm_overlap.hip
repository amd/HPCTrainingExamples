#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


/* Macro for checking GPU API return values */
#define HIP_ASSERT(call)                                                                        \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)


// HIP kernel. Each thread takes care of one element of input
__global__ void cube(double *input, double *output,int offset)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x+offset;
    for (size_t i = 0; i < 1000; ++i)
      output[id] = input[id]*input[id]*input[id];
      

}
 
void usage()
{
  printf("Usage: ./compute_comm_overlap <nstreams>\n");
  exit(1);
  return;
}

int main( int argc, char* argv[] )
{
    // Read input arg
    if (argc < 2)
      usage();
    int num_streams = atoi(argv[1]);

    // Size of vectors
    int n = 100000000;
    int elements_per_stream = n/num_streams;
    int bytes_per_stream = elements_per_stream*sizeof(double);


    // Host input vectors
    double *h_input1;
    //Host output vector
    double *h_output1;
    //Host output vector for verification
    double *h_verify1;
 
    // Device input vectors
    double *d_input1;
    //Device output vector
    double *d_output1;

    //Creating events for timers
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    //Creating streams

    hipStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
    {
       HIP_ASSERT(hipStreamCreate(&streams[i]));	    
    }

   
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
  
    // Allocate page locked memory for these vectors on host
    HIP_ASSERT(hipHostMalloc(&h_input1, bytes));
    HIP_ASSERT(hipHostMalloc(&h_output1, bytes));

   
    h_verify1 = (double*)malloc(bytes);
   

   printf("Finished allocating vectors on the CPU\n");     
    // Allocate memory for each vector on GPU
   HIP_ASSERT(hipMalloc(&d_input1, bytes));
   HIP_ASSERT(hipMalloc(&d_output1, bytes));
 
   printf("Finished allocating vectors on the GPU\n");

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_input1[i] = sin(i);
    }


    int blockSize, gridSizePerStream;

    // Number of threads in each thread block
    blockSize = 256;

    // Number of thread blocks in grid
    gridSizePerStream = (int)ceil((float)elements_per_stream/blockSize);


    hipEventRecord(start);
    // Copy and execute in loop
   
#define SPLIT_DATACOPY_KERNEL_CALLS
#ifndef SPLIT_DATACOPY_KERNEL_CALLS
    for(int i = 0; i < num_streams; i++)
    {
      int offset = i * elements_per_stream;
      HIP_ASSERT(hipMemcpyAsync(&d_input1[offset],&h_input1[offset],bytes_per_stream,hipMemcpyHostToDevice,streams[i]));
      cube<<<gridSizePerStream,blockSize,0,streams[i]>>>(d_input1,d_output1,offset);
      HIP_ASSERT(hipMemcpyAsync(&h_output1[offset],&d_output1[offset], bytes_per_stream, hipMemcpyDeviceToHost,streams[i]));
    }
#else
    // split H2D copies and kernel calls into separate loops
    for(int i = 0; i < num_streams; i++)
    {
      int offset = i * elements_per_stream;
      HIP_ASSERT(hipMemcpyAsync(&d_input1[offset],&h_input1[offset],bytes_per_stream,hipMemcpyHostToDevice,streams[i]));
    }
    for(int i = 0; i < num_streams; i++)
    {
      int offset = i * elements_per_stream;
      cube<<<gridSizePerStream,blockSize,0,streams[i]>>>(d_input1,d_output1,offset);
    }
    for(int i = 0; i < num_streams; i++)
    {
      int offset = i * elements_per_stream;
      HIP_ASSERT(hipMemcpyAsync(&h_output1[offset],&d_output1[offset], bytes_per_stream, hipMemcpyDeviceToHost,streams[i]));
    }
#endif //SPLIT_DATACOPY_KERNEL_CALLS

    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    printf("Time required total %f\n", milliseconds);
    printf("Finished copying the output vector from the GPU to the CPU\n");

   //Compute for CPU 
    for(i=0; i <n; i++)
    {
    h_verify1[i] = h_input1[i] * h_input1[i] * h_input1[i];
    }


    //Verfiy results
    for(i=0; i <n; i++)
    {
    if (abs(h_verify1[i] - h_output1[i]) > 1e-5) 
     {
     printf("Error at position i %d, Expected: %f, Found: %f \n", i, h_verify1[i], h_output1[i]);
     }  
    }

    printf("Releasing GPU memory\n");
     
    // Release device memory
    HIP_ASSERT(hipFree(d_input1));
    HIP_ASSERT(hipFree(d_output1));
    
    // Release host memory
    printf("Releasing CPU memory\n");
    HIP_ASSERT(hipHostFree(h_input1));
    HIP_ASSERT(hipHostFree(h_output1));
    free(h_verify1);
  
    return 0;
}

