#pragma once

#include "helper.hpp"

#define BLOCK_DIM_X 64
#define BLOCK_DIM_Y 8


template <int R>
__launch_bounds__((BLOCK_DIM_X) * (BLOCK_DIM_Y))
__global__ void compute_fd_z_kernel(float *__restrict__ p_out, const float *__restrict__ p_in, 
                                        const float *__restrict__ d, int line, int
        slice, int x0, int x1, int y0, int y1, int z0, int z1) {

    const int i = x0 + threadIdx.x + blockIdx.x * blockDim.x;
    const int j = y0 + threadIdx.y + blockIdx.y * blockDim.y;
    const int k = z0 + threadIdx.z + blockIdx.z * blockDim.z;

    if (i >= x1 || j >= y1 || k >= z1) return;

    size_t pos = i + line * j + slice * k;

    // Shift pointers such that that p_in points to the first value in the stencil
    p_in += pos - R * slice;
    p_out += pos;
     
    // Compute the finite difference approximation
    float out = 0.0f;
    for (int r = 0; r <= 2 * R; ++r) {
        out += p_in[0] * d_dz<R>[r]; 
        p_in += slice;
    }

    // Write the result
    p_out[0] = out;

}


template <int R>
void compute_fd_z(float *p_out, const float *p_in, const float *d, int line, int
        slice, int x0, int x1, int y0, int y1, int z0, int z1) {
    /* Computes a central high order finite difference (FD) approximation in the z-direction
     * The computation is applied for all grid points in x0 <= i < x1, y0 <= j < y1, z0 <= k < z1
     
     p_out: Array to write the result to
     p_in: Array to apply the approximation to
     d: Array of length 2 * R + 1 that contains the finite difference approximations, including scaling by grid spacing. 
     R: Stencil radius
     stride: The stride to use for the stencil. This parameter controls the direction in which the stencil is applied.
    */


    dim3 block (BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid;
    grid.x = ceil(x1 - x0, block.x);
    grid.y = ceil(y1 - y0, block.y);
    grid.z = ceil(z1 - z0, block.z);

    compute_fd_z_kernel<R><<<grid, block>>>(p_out, p_in, d, line, slice, x0, x1, y0, y1, z0,
            z1);
    HIP_CHECK(hipGetLastError());
     
}

#undef BLOCK_DIM_X
#undef BLOCK_DIM_Y
