#pragma once

#include "helper.hpp"

#define BLOCK_DIM_X 64 * (4 / RADIUS)
#define BLOCK_DIM_Y RADIUS

template <int R>
__launch_bounds__((BLOCK_DIM_X) * (BLOCK_DIM_Y))
__global__ void compute_fd_xyz_lds_window_vec_kernel(float *p_out, const float *p_in, int line, int
        slice, int x0, int x1, int y0, int y1, int z0, int z1, int dimz) {
    
    const size_t i = (x0 + VEC_LEN * (threadIdx.x + blockIdx.x * blockDim.x));
    const size_t j = y0 + threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= x1) return; 

    // Determine the k indices covered by this sliding window
    // The extent cannot exceed the z1
    const int kbegin = z0 + blockIdx.z * dimz;
    const int kend = kbegin + dimz > z1 ? z1 : kbegin + dimz;

    size_t pos = i + line * j + slice * kbegin;
    size_t slice_vec = slice >> VEC_EXP;
    size_t line_vec = line >> VEC_EXP;

    // Shift pointers such that that p_in points to the first value in the sliding window
    p_in += pos - R * slice;
    p_out += pos;
    const vec *p_in_vec = reinterpret_cast<const vec*>(p_in);
    vec *p_out_vec = reinterpret_cast<vec*>(p_out);
     
     // LDS for y direction
    const int lds_y = BLOCK_DIM_Y + 2*R;
    const int sj = y0 + threadIdx.y;
    size_t spos = threadIdx.x + sj * BLOCK_DIM_X;
    __shared__ vec smem[BLOCK_DIM_X * lds_y];
    
    // z direction sliding window
    vec w[2 * R + 1];

    // solution register
    vec out[R+1];

    // x direction stencil
    float x_win[2 * XWIN + VEC_LEN];
    vec *x_win_vec = reinterpret_cast<vec*>(x_win);

    // 1. Prime the z sliding window
    for (int r = 0; r < R; ++r) {
        w[r] = p_in_vec[0]; 
        p_in_vec += slice_vec;
    }
    for (int r = R; r < 2 * R; ++r) {
        
        // 2. Load x into registers
        for (int r2 = 0; r2 < 2*XWIN_VEC + 1; ++r2) 
            x_win_vec[r2] = p_in_vec[0 - XWIN_VEC + r2]; 
        
        // 3. Load y into LDS
        __syncthreads();
        {
            smem[spos - (BLOCK_DIM_X * R)] = p_in_vec[0 - R * line_vec];
            smem[spos] = x_win_vec[XWIN_VEC];
            smem[spos + (BLOCK_DIM_X * BLOCK_DIM_Y)] = p_in_vec[0 + line_vec * BLOCK_DIM_Y];
        }
        __syncthreads();
        
        // 4. Compute xy stencils
        out[r-R] = {0.0f};
        for (int r2 = 0; r2 <= 2 * R; ++r2) {
            out[r-R] += smem[spos + (r2 - R) * BLOCK_DIM_X] * d_dy<R>[r2]; // y-direction
            for (int ii = 0; ii < VEC_LEN; ++ii)
                out[r-R][ii] += x_win[XWIN_OFF + r2 + ii] * d_dx<R>[r2]; // x-direction
        }

        // Prime the z sliding window
        w[r] = x_win_vec[XWIN_VEC]; 
        p_in_vec += slice_vec;
    }
    
    // Apply the sliding window along the given grid direction
    for (int k = kbegin; k < kend; ++k) {

        // 2. Load x into registers
        for (int r2 = 0; r2 < 2*XWIN_VEC+1; ++r2)
            x_win_vec[r2] = p_in_vec[0 - XWIN_VEC + r2]; // x - R
        
        // 3. Load y into LDS
        __syncthreads();
        {
            smem[spos - (BLOCK_DIM_X * R)] = p_in_vec[0 - R * line_vec]; // y - R
            smem[spos] = x_win_vec[XWIN_VEC];
            smem[spos + (BLOCK_DIM_X * BLOCK_DIM_Y)] = p_in_vec[0 + line_vec * BLOCK_DIM_Y]; // y + R
        }
        __syncthreads();
        
        // 4. Compute xyz stencils
        w[2*R] = x_win_vec[XWIN_VEC]; 
        out[R] = {0.0f};
        for (int r = 0; r <= 2 * R; ++r) {
            out[0] += w[r] * d_dz<R>[r]; // z-direction
            out[R] += smem[spos + (r - R) * BLOCK_DIM_X] * d_dy<R>[r]; // y-direction
            for (int ii = 0; ii < VEC_LEN; ++ii)
                out[R][ii] += x_win[XWIN_OFF + r + ii] * d_dx<R>[r]; // x-direction
        }

        // 5. Write only if within y boundary
        if (j < y1)
            ntstore(p_out_vec[0],out[0]);
        
        // 6. Update the sliding window by shifting it forward one step
        for (int r = 0; r < R; ++r)
            out[r] = out[r+1];
        for (int r = 0; r < 2*R; ++r)
            w[r] = w[r+1];
        
        
        // Increment pointers
        p_in_vec += slice_vec;
        p_out_vec += slice_vec;
    }
}

template <int R>
void compute_fd_xyz_lds_window_vec(float *p_out, const float *p_in, const float *d, int line, int
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
    //size_t remainder = (y1 - y0) % R;
    grid.x = ceil(x1 - x0, VEC_LEN * block.x);
    grid.y = ceil(y1 - y0, block.y);
    grid.z = ceil(z1 - z0, dimwin);

    compute_fd_xyz_lds_window_vec_kernel<R><<<grid, block>>>(p_out, p_in, line, slice, x0, x1,
            y0, y1, z0, z1, dimwin);
    HIP_CHECK(hipGetLastError());
     
}

#undef BLOCK_DIM_X
#undef BLOCK_DIM_Y
