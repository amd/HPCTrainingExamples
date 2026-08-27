/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#include "Jacobi.hpp"

//Jacobi iterative method
// U = U + D^{-1}*(RHS - AU)
__global__ void JacobiIterationKernel(const int N,
                                 const dfloat inv_dxy_factor,
                                 const dfloat *__restrict__ RHS,
                                 const dfloat *__restrict__ AU,
                                       dfloat *__restrict__ RES,
                                       dfloat *__restrict__ U) {

  const int id = threadIdx.x+blockIdx.x*blockDim.x;

  if (id<N) {
    dfloat r_res = RHS[id] - AU[id];
    RES[id] = r_res;
    U[id] += r_res*inv_dxy_factor;
  }
}

void JacobiIteration(grid_t& grid, mesh_t& mesh,
                     hipStream_t stream,
                     dfloat* d_RHS,
                     dfloat* d_AU,
                     dfloat* d_RES,
                     dfloat* d_U) {

  int xthreads = 512;

  dim3 threads(xthreads,1,1);
  dim3 blocks((mesh.N+xthreads-1)/xthreads, 1, 1);

  hipLaunchKernelGGL(JacobiIterationKernel,
                     blocks,
                     threads,
                     0, stream,
                     mesh.N,
                     mesh.inv_dxy_factor,
                     d_RHS, d_AU, d_RES, d_U);
}
