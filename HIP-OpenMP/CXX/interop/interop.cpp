#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <hip/hip_runtime.h>
#include <iostream>

/* Macro for checking GPU API return values */
#define hipCheck(call)                                                                        \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)


void hip_square(const int gridSizePerStream, const int blockSize, double* input, const int elements, hipStream_t stream);

int main( int argc, char* argv[] )
{
    int blockSize = 64;
    int gridSize = 228; 
    printf("gridSize: %d\n", gridSize);
    printf("blockSize: %d\n", blockSize);

    // Size of vectors
    int elements = 100000000;

    // Host input vectors
    double *h_input;
    //Host output vector for verification
    double *h_verify;
 
    // Device input vectors
    double *d_input;

    //Creating events for timers
    hipEvent_t start, stop;
    hipCheck( hipEventCreate(&start) );
    hipCheck( hipEventCreate(&stop) ) ;

    // Size, in bytes, of each vector
    size_t bytes = elements*sizeof(double);
  
    // Allocate page locked memory for these vectors on host
    hipCheck( hipHostMalloc(&h_input, bytes) );
   
    h_verify = (double*)malloc(bytes);
   
   printf("Finished allocating vectors on the CPU\n");     
    // Allocate memory for each vector on GPU
   hipCheck( hipMalloc(&d_input, bytes) );
 
   printf("Finished allocating vectors on the GPU\n");

    int i;
    // Initialize vectors on host
    for( i = 0; i < elements; i++ ) {
        h_input[i] = sin(i);
        h_verify[i] = h_input[i] * h_input[i] + M_PI;
    }

    omp_interop_t iobj = omp_interop_none;
    #pragma omp interop init(targetsync: iobj)
    hipStream_t stream = (hipStream_t) omp_get_interop_ptr(iobj, omp_ipr_targetsync, NULL);
    hipCheck( hipStreamCreate(&stream) );

    hipCheck( hipEventRecord(start) );

    // Copy and execute in loop
   
    hipCheck( hipMemcpyAsync(d_input, h_input, bytes, hipMemcpyHostToDevice, stream) );
    hip_square(gridSize, blockSize, d_input, elements, stream);
    hipCheck( hipStreamSynchronize(stream) );
    // interop use does not seem to work
    // #pragma omp interop use(iobj)
    #pragma omp target teams loop 
    for( i = 0; i < elements; i++ ) {
       double tmp;
       // Just to make the kernel take longer
       for (int j=0; j<100000; j++){
          tmp = M_PI;
       }
       d_input[i] += tmp;
    }
    hipCheck( hipMemcpyAsync(h_input, d_input, bytes , hipMemcpyDeviceToHost, stream) );

    #pragma omp interop destroy(iobj)

    hipCheck( hipEventRecord(stop) );
    hipCheck( hipEventSynchronize(stop) );

    float milliseconds = 0;
    hipCheck( hipEventElapsedTime(&milliseconds, start, stop) );
    printf("Time required total (ms) %f\n", milliseconds);
    printf("Finished copying the output vector from the GPU to the CPU\n");

    //Verfiy results
    int no_error = 1;
    for(i=0; i <elements; i++){
       if (abs(h_verify[i] - h_input[i]) > 1e-12) {
          printf("Error at position i %d, Expected: %f, Found: %f \n", i, h_verify[i], h_input[i]);
          no_error = 0;
          break;
       }  
    }

    printf("Releasing GPU memory\n");
     
    // Release device memory
    hipCheck(hipFree(d_input));
    
    // Release host memory
    printf("Releasing CPU memory\n");
    hipCheck(hipHostFree(h_input));
    free(h_verify);

    if(no_error){
       printf("PASS!\n");
    }
    else{
       printf("FAIL!\n");
    }
    
    return 0;
}

