#pragma once

#include "helper.hpp"

#define BLOCK_DIM_X 64
#define BLOCK_DIM_Y 8



__launch_bounds__((BLOCK_DIM_X) * (BLOCK_DIM_Y))
__global__ void max_absval_gpu_kernel(float *out, const float *u, const float *v, int line, int
        slice, int x0, int x1, int y0, int y1, int z0, int z1, int dimz) {

    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < x0 || i >= x1 || j < y0 || j >= y1) return;

    const int kbegin = z0 + blockIdx.z * dimz;
    const int kend = kbegin + dimz > z1 ? z1 : kbegin + dimz;

    size_t pos = POS(i,j,kbegin);

    u += pos;
    v += pos;

    float maxval = 0.0f;

    for (int k = kbegin; k < kend; ++k) {

        float diff = fabs(u[0] - v[0]);

        maxval = maxval < diff ? diff : maxval;

        // Increment pointers
        u += slice;
        v += slice;
        
    }

    atomicMax(out, maxval);

}

float max_absval_gpu(float *u, const float *v, int line, int slice, int x0, int x1, int y0, int y1, int z0, int z1, int dimz=-1) {
    /* Computes the absolute maximum value difference in x0 <= i < x1, y0 <= j < y1, z0 <= k < z1.
     
     p_out: Array to write the result to
     p_in: Array to apply the approximation to
     d: Array of length 2 * R + 1 that contains the finite difference approximations, including scaling by grid spacing. 
     R: Stencil radius
     stride: The stride to use for the stencil. This parameter controls the direction in which the stencil is applied.
     dimz: Number of grid points to procees in the z-direction
    */


    dimz = dimz == -1 ? z1 - z0 : dimz;

    dim3 block (BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid;
    size_t stride;
    grid.x = ceil(x1, block.x);
    grid.y = ceil(y1, block.y);
    grid.z = ceil(z1, dimz);

    float *d_out; 
    HIP_CHECK( hipMalloc(&d_out, sizeof(float)) );

    max_absval_gpu_kernel<<<grid, block>>>(d_out, u, v, line, slice, x0, x1, y0, y1, z0, z1, dimz);
    HIP_CHECK(hipGetLastError());
    float *out = new float[1];

    HIP_CHECK(hipMemcpy(out, d_out, sizeof(float), hipMemcpyDeviceToHost));
    return out[0];
     
}

#undef BLOCK_DIM_X
#undef BLOCK_DIM_Y
