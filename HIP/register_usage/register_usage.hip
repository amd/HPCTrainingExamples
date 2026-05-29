#include <hip/hip_runtime.h>

/* Macro for checking GPU API return values */
#define HIP_ASSERT(call)                                                                        \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)

template<int REG_PRESSURE>
__global__ void register_heavy_kernel(
    const int* input, int* output, int param1, int param2, int param3, int param4, int n){

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tid < n) {
    
       int scalar_vars[REG_PRESSURE];
   
#if SGPR 
          scalar_vars[0] = param1;
          scalar_vars[1] = param2;
          scalar_vars[2] = param3;
          scalar_vars[3] = param4;
#endif


#if VGPR
          scalar_vars[0] = tid;
          scalar_vars[1] = tid+1;
          scalar_vars[2] = tid+2;
          scalar_vars[3] = tid+3;
#endif
    
       #pragma unroll
       for (int i = 4; i < REG_PRESSURE; i++) {
          scalar_vars[i] = scalar_vars[i-1] + scalar_vars[i-2];
       }
    
       int scalar_accum = 0;
    
       #pragma unroll
       for (int i = 0; i < REG_PRESSURE; i++) {
           scalar_accum += scalar_vars[i];
       }
    
       #pragma unroll
       for (int i = 0; i < REG_PRESSURE - 1; i++) {
#if SGPR 
          scalar_vars[i] = scalar_vars[i] * scalar_vars[i+1] + scalar_accum;
#endif
#if VGPR 
          scalar_vars[i] = scalar_vars[i] * scalar_vars[i+1] + tid;
#endif
       }
    
       int scalar_result = 0;
       #pragma unroll
       for (int i = 0; i < REG_PRESSURE; i++) {
           scalar_result ^= scalar_vars[i];
           scalar_result = (scalar_result << 1) | (scalar_result >> 31);
       }
    
       int scalar_mult = scalar_result * scalar_accum;
       int scalar_div = (scalar_mult / 17) + 1;
       int scalar_mod = scalar_mult % 23;
       int scalar_final = scalar_div + scalar_mod;
    
       int input_val = input[tid];
       output[tid] = input_val + scalar_final;
   }
}

// Main function
int main(int argc, char* argv[]) {
   
    printf("\n");
    printf("This example shows how the implementation of a kernel \n");
    printf("can affect resource utilization at the register level \n");
 
    int n = 1024 * 1024;
    int block_size = 256;
    int grid_size = ceil(float(n)) / block_size;
    printf("\n");
    printf("Array size: %d\n", n);
    printf("Grid size: %d\n", grid_size);
    printf("Block size: %d\n", block_size);
    
    int* h_input = new int[n];
    int* h_output = new int[n];
    
    for (int i = 0; i < n; i++) {
        h_input[i] = i % 1000;
    }
    
    int *d_input, *d_output;
    HIP_ASSERT(hipMalloc(&d_input, n * sizeof(int)));
    HIP_ASSERT(hipMalloc(&d_output, n * sizeof(int)));
    
    HIP_ASSERT(hipMemcpy(d_input, h_input, n * sizeof(int), hipMemcpyHostToDevice));
    
    int param1 = 10;
    int param2 = 20;
    int param3 = 30;
    int param4 = 40;
    
    // Test with different register pressure levels
#if SGPR
    printf("\n--- Considering a kernel implementation that is increasingly using more SGPRs \n");
#endif

#if VGPR
    printf("\n--- Considering a kernel implementation that is increasingly using more VGPRs \n");
#endif
    
    printf("\nLaunching kernel with REG_PRESSURE = 8\n");
    register_heavy_kernel<8><<<grid_size,block_size>>>(d_input, d_output, 
                       param1, param2, param3, param4, n);
    HIP_ASSERT(hipDeviceSynchronize());
    
    printf("Launching kernel with REG_PRESSURE = 16\n");
    register_heavy_kernel<16><<<grid_size,block_size>>>(d_input, d_output, 
                       param1, param2, param3, param4, n);
    HIP_ASSERT(hipDeviceSynchronize());
    
    printf("Launching kernel with REG_PRESSURE = 32\n");
    register_heavy_kernel<32><<<grid_size,block_size>>>(d_input, d_output, 
                       param1, param2, param3, param4, n);
    HIP_ASSERT(hipDeviceSynchronize());
    
    printf("Launching kernel with REG_PRESSURE = 64\n");
    register_heavy_kernel<64><<<grid_size,block_size>>>(d_input, d_output, 
                       param1, param2, param3, param4, n);
    HIP_ASSERT(hipDeviceSynchronize());
    printf("\n");

    printf("Inspect the compilation output to see how the above kernel launches \n");
#if SGPR
    printf("make different use of the scalar general purpose registers (SGPRs). \n");
    printf("Compile with 'make VGPR=1' the to see the effect on the \n");
    printf("vector general purpose registers (VGPRs) instead. \n");
#endif
#if VGPR
    printf("make different use of the vector general purpose registers (VGPRs). \n");
#endif
    printf("\n");
}
