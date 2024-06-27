/******************************************************************************
Copyright (c) 2022 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
******************************************************************************/

// Kernel 2 - Simple loop tiling in Y dimension (inefficient)
#define KERNEL 2
#include "hip/hip_runtime.h"

// Tiling factor
#define m 8
template <typename T>
__global__ void laplacian_kernel(T * f, const T * u, int nx, int ny, int nz, T invhx2, T invhy2, T invhz2, T invhxyz2) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = m*(threadIdx.y + blockIdx.y * blockDim.y);
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // Exit if this thread is on the xz boundary
    if (i == 0 || i >= nx - 1 ||
        k == 0 || k >= nz - 1)
        return;

    const int slice = nx * ny;
    size_t pos = i + nx * j + slice * k;
    
    // Each thread accumulates m stencils in the y direction
    T Lu[m] = {0};
    
    // Scalar for reusable data
    T center;

    // Loop tiling
    for (int n = 0; n < m; n++) {
      center = u[pos + n*nx]; // store for reuse
      
      // x direction
      Lu[n] += center *invhxyz2
            + (u[pos - 1 + n*nx] +  u[pos + 1 + n*nx]) * invhx2;
      
      // y - 1, first n
      if (n == 0) Lu[n] += u[pos - nx + n*nx] * invhy2;
      
      // reuse: y + 1 for prev n
      if (n > 0) Lu[n-1] += center * invhy2;
      
      // reuse: y - 1 for next n
      if (n < m - 1) Lu[n+1] += center * invhy2;
      
      // y + 1, last n
      if (n == m - 1) Lu[n] += u[pos + nx + n*nx] * invhy2;
      
      // z - 1, center, z + 1
      Lu[n] += (u[pos - slice + n*nx] + u[pos + slice + n*nx]) * invhz2;
    }

    // Store only if thread is inside y boundary
    for (int n = 0; n < m; n++)
      if (j + n > 0 && j + n < ny - 1)
        f[pos + n*nx] = Lu[n];
} 

template <typename T>
void laplacian(T *d_f, T *d_u, int nx, int ny, int nz, int BLK_X, int BLK_Y, int BLK_Z, T hx, T hy, T hz) {

    dim3 block(BLK_X, BLK_Y, BLK_Z);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / (block.y * m) + 1, (nz - 1) / block.z + 1);
    T invhx2 = (T)1./hx/hx;
    T invhy2 = (T)1./hy/hy;
    T invhz2 = (T)1./hz/hz;
    T invhxyz2 = -2. * (invhx2 + invhy2 + invhz2);

    laplacian_kernel<<<grid, block>>>(d_f, d_u, nx, ny, nz, invhx2, invhy2, invhz2, invhxyz2);
} 
