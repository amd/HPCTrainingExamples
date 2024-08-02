#pragma once

#include "helper.hpp"

#define BLOCK_DIM_X 64 * (4 / RADIUS)
#define BLOCK_DIM_Y RADIUS

template <int R>
__launch_bounds__((BLOCK_DIM_X) * (BLOCK_DIM_Y))
__global__ void compute_fd_xy_lds_vec_kernel(float *p_out, const float *p_in, const float *d, int line, int
        slice, int x0, int x1, int y0, int y1, int z0, int z1) {

    const int sj = y0 + threadIdx.y;
    const int lds_y = BLOCK_DIM_Y + 2*R;
    const int i = x0 + VEC_LEN*(threadIdx.x + blockIdx.x * blockDim.x);
    const int j = sj + blockIdx.y * blockDim.y;
    const int k = z0 + threadIdx.z + blockIdx.z * blockDim.z;
    size_t pos = i + line * j + slice * k;
    size_t spos = threadIdx.x + sj * BLOCK_DIM_X;
    size_t line_vec = line >> VEC_EXP;

    p_in += pos;
    p_out += pos;
    float x_win[2 * XWIN + VEC_LEN] = {0.0f};
    __shared__ vec smem[BLOCK_DIM_X * lds_y];

    // Recast as vectorized floats
    const vec *p_in_vec = reinterpret_cast<const vec*>(p_in);
    vec *p_out_vec = reinterpret_cast<vec*>(p_out);
    vec *x_win_vec = reinterpret_cast<vec*>(x_win);

    // Read x into registers
    for (int r = 0; r < 2*XWIN_VEC+1; ++r)
        x_win_vec[r] = p_in_vec[0 - XWIN_VEC + r];
    
    // Read y into LDS
    smem[spos] = x_win_vec[XWIN_VEC];
    smem[spos - (BLOCK_DIM_X * R)] = p_in_vec[0 - R*line_vec];
    smem[spos + BLOCK_DIM_X * BLOCK_DIM_Y] = p_in_vec[0 + line_vec * BLOCK_DIM_Y];
    __syncthreads();

    if (i >= x1 || j >= y1 || k >= z1) return;
    
    // Compute the finite difference approximation in the xy-direction
    vec out = {0.0f};
    for (int r = 0; r <= 2 * R; ++r) {
        out += smem[spos + (r - R) * BLOCK_DIM_X] * d_dy<R>[r]; 
        for (int ii = 0; ii < VEC_LEN; ++ii)
            out[ii] += x_win[XWIN_OFF + r + ii] * d_dx<R>[r];
    }
    
    ntstore(p_out_vec[0],out);

}

template <int R>
void compute_fd_xy_lds_vec(float *p_out, const float *p_in, const float *d, int line, int
        slice, int x0, int x1, int y0, int y1, int z0, int z1) {
    /* Computes a central high order finite difference (FD) approximation in the x and y-directions 
     * using overlapping thread blocks in the x-direction and LDS for reducing global memory loads. 
     * The computation is applied for all grid points in x0 <= i < x1, y0 <= j < y1, z0 <= k < z1
     
     p_out: Array to write the result to
     p_in: Array to apply the approximation to
     d: Array of length 2 * R + 1 that contains the finite difference approximations, including scaling by grid spacing. 
     R: Stencil radius
    */


    dim3 block (BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid;
    grid.x = ceil((x1 - x0), VEC_LEN * block.x);
    grid.y = ceil(y1 - y0, block.y);
    grid.z = ceil(z1 - z0, block.z);

    compute_fd_xy_lds_vec_kernel<R><<<grid, block>>>(p_out, p_in, d, line, slice, x0, x1,
            y0, y1, z0, z1);
    HIP_CHECK(hipGetLastError());
     
}

#undef BLOCK_DIM_X
#undef BLOCK_DIM_Y
