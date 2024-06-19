/******************************************************************************
Copyright (c) 2023 Advanced Micro Devices, Inc. (AMD)

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

// Kernel 6 - Grid launch configuration
#define KERNEL 6
#include "hip/hip_runtime.h"

// Tiling factor
#define m 8
// Launch bounds
#define LB 256
template <typename T>
__launch_bounds__(LB)
__global__ void laplacian_kernel(T * f, const T * u, int nx, int ny, int nz, T invhx2, T invhy2, T invhz2, T invhxyz2) {
    
    int i = threadIdx.x + blockIdx.z * blockDim.x;
    int j = m*(threadIdx.y + blockIdx.x * blockDim.y);
    int k = threadIdx.z + blockIdx.y * blockDim.z;
    
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
    
    // z - 1, loop tiling
    for (int n = 0; n < m; n++)
        Lu[n] += u[pos - slice + n*nx] * invhz2;
    
    // y - 1
    Lu[0]   += j > 0 ? u[pos - 1*nx] * invhy2 : 0; // bound check
    
    // x direction, loop tiling
    for (int n = 0; n < m; n++) {
        // x - 1
        Lu[n] += u[pos - 1 + n*nx] * invhx2;

        // x
        center = u[pos + n*nx]; // store for reuse
        Lu[n] += center * invhxyz2;

        // x + 1
        Lu[n] += u[pos + 1 + n*nx] * invhx2;
        
        // reuse: y + 1 for prev n
        if (n > 0) Lu[n-1] += center * invhy2;

        // reuse: y - 1 for next n
        if (n < m - 1) Lu[n+1] += center * invhy2;
    }
    
    // y + 1
    Lu[m-1]  += j < ny - m ? u[pos + m*nx] * invhy2 : 0; // bound check
    
    // z + 1, loop tiling
    for (int n = 0; n < m; n++)
      Lu[n] += u[pos + slice + n*nx] * invhz2;

    // Store only if thread is inside y boundary
    for (int n = 0; n < m; n++)
      if (n + j > 0 && n + j < ny - 1)
        __builtin_nontemporal_store(Lu[n],&f[pos + n*nx]);
} 

template <typename T>
void laplacian(T *d_f, T *d_u, int nx, int ny, int nz, int BLK_X, int BLK_Y, int BLK_Z, T hx, T hy, T hz) {

    dim3 block(BLK_X, BLK_Y, BLK_Z);
    dim3 grid((ny - 1) / (block.y * m) + 1, (nz - 1) / block.z + 1, (nx - 1) / block.x + 1);
    T invhx2 = (T)1./hx/hx;
    T invhy2 = (T)1./hy/hy;
    T invhz2 = (T)1./hz/hz;
    T invhxyz2 = -2. * (invhx2 + invhy2 + invhz2);

    laplacian_kernel<<<grid, block>>>(d_f, d_u, nx, ny, nz, invhx2, invhy2, invhz2, invhxyz2);
} 
