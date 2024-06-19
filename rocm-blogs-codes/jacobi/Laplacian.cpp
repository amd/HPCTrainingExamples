//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include "Jacobi.hpp"

// AU_i,j = (-U_i+1,j + 2U_i,j - U_i-1,j)/dx^2 +
//          (-U_i,j+1 + 2U_i,j - U_i,j-1)/dy^2
#ifdef _HIP
__global__
__launch_bounds__(BLOCK_SIZE)
void LaplacianKernel(const int localNx,
                     const int localNy,
                     const int stride,
                     const dfloat fac_dx2,
                     const dfloat fac_dy2,
                     const dfloat * U,
                     dfloat * AU)
{
  int tid = GET_GLOBAL_ID_0;
  if (tid > localNx + localNy * stride || tid < stride + 1)
      return;

  const int tid_l = tid - 1;
  const int tid_r = tid + 1;
  const int tid_d = tid - stride;
  const int tid_u = tid + stride;

  __builtin_nontemporal_store((-U[tid_l] + 2*U[tid] - U[tid_r])*fac_dx2 +
            (-U[tid_d] + 2*U[tid] - U[tid_u])*fac_dy2, &(AU[tid]));
}
#endif

void Laplacian(mesh_t& mesh,
               const dfloat _1bydx2,
               const dfloat _1bydy2,
               dfloat* U,
               dfloat* AU) 
{
  int stride = mesh.Nx;
  int localNx = mesh.Nx-2;
  int localNy = mesh.Ny-2;
#ifdef _HIP
  LaplacianKernel<<<mesh.grid,mesh.block>>>(localNx, localNy, stride, _1bydx2, _1bydy2, U, AU);
#else
  #pragma omp target teams distribute parallel for collapse(2)
  for (int j=0;j<localNy;j++) {
    for (int i=0;i<localNx;i++) {

      const int id = (i+1) + (j+1)*stride;

      const int id_l = id - 1;
      const int id_r = id + 1;
      const int id_d = id - stride;
      const int id_u = id + stride;

       AU[id] = (-U[id_l] + 2*U[id] - U[id_r])*_1bydx2 +
                (-U[id_d] + 2*U[id] - U[id_u])*_1bydy2;
    }
  }
#endif
}
