#pragma once

#include "helper.hpp"

#define BLOCK_DIM_X 64
#define BLOCK_DIM_Y 4

template <int R>
__launch_bounds__((BLOCK_DIM_X) * (BLOCK_DIM_Y))
__global__ void compute_fd_z_window_kernel(float *p_out, const float *p_in, const float *d, int line, int
        slice, int x0, int x1, int y0, int y1, int z0, int z1, int dimz) {

    const int i = x0 + threadIdx.x + blockIdx.x * blockDim.x;
    const int j = y0 + threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= x1 || j >= y1) return;

    // Determine the k indices covered by this sliding window
    // The extent cannot exceed the z1
    const int kbegin = z0 + blockIdx.z * dimz;
    const int kend = kbegin + dimz > z1 ? z1 : kbegin + dimz;

    // Sliding window
    float w[2 * R + 1];

    size_t pos = i + line * j + slice * kbegin;

    // Shift pointers such that that p_in points to the first value in the sliding window
    p_in += pos - R * slice;
    p_out += pos;
     
    // 1. Prime the sliding window
    for (int r = 0; r < 2 * R; ++r) {
        w[r] = p_in[0]; 
        p_in += slice;
    }
    
    // Apply the sliding window along the given grid direction
    for (int k = kbegin; k < kend; ++k) {
        // 2. Load the next value into the sliding window at its last position
        w[2 * R] = p_in[0];

        // 3. Compute the finite difference approximation using the sliding window
        float out = 0.0f;
        for (int r = 0; r <= 2 * R; ++r)
            out += w[r] * d_dz<R>[r]; 
        
        p_out[0] = out;

        // 4. Update the sliding window by shifting it forward one step
        for (int r = 0; r < 2 * R; ++r)
            w[r] = w[r+1];

        // Increment pointers
        p_in += slice;
        p_out += slice;
    }
}


template <int R>
void compute_fd_z_window(float *p_out, const float *p_in, const float *d, int line, int
        slice, int x0, int x1, int y0, int y1, int z0, int z1, int dimwin=-1) {
    /* Computes a central high order finite difference (FD) approximation in the z-direction using a sliding window 
     * The computation is applied for all grid points in x0 <= i < x1, y0 <= j < y1, z0 <= k < z1
     
     p_out: Array to write the result to
     p_in: Array to apply the approximation to
     d: Array of length 2 * R + 1 that contains the finite difference approximations, including scaling by grid spacing. 
     R: Stencil radius
     stride: The stride to use for the stencil. This parameter controls the direction in which the stencil is applied.
     dimwin: Number of grid points to cover in a sliding window
    */


    dimwin = dimwin == -1 ? z1 - z0 : dimwin;

    dim3 block (BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid;
    size_t stride;
    grid.x = ceil(x1 - x0, block.x);
    grid.y = ceil(y1 - y0, block.y);
    grid.z = ceil(z1 - z0, dimwin);

    compute_fd_z_window_kernel<R><<<grid, block>>>(p_out, p_in, d, line, slice, x0, x1,
            y0, y1, z0, z1, dimwin);
    HIP_CHECK(hipGetLastError());
     
}

#undef BLOCK_DIM_X
#undef BLOCK_DIM_Y
