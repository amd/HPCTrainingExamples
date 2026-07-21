#include <iostream>
#include <hip/hip_runtime.h>

#define HIP_CHECK(stat)                                           \
{                                                                 \
    if(stat != hipSuccess)                                        \
    {                                                             \
        std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
        " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(-1);                                                 \
    }                                                             \
}

__global__ void vector_add(double *a, double *b, int n)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride)
        b[i] = a[i] + b[i];
}

extern "C"
void hip_function(double *a, double *b, int n)
{
    vector_add<<<n/256, 256>>>(a, b, n);
    HIP_CHECK(hipDeviceSynchronize());
}
