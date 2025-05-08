//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include "Jacobi.hpp"

// AU_i,j = (-U_i+1,j + 2U_i,j - U_i-1,j)/dx^2 +
//          (-U_i,j+1 + 2U_i,j - U_i,j-1)/dy^2

void Laplacian(mesh_t& mesh,
               const dfloat _1bydx2,
               const dfloat _1bydy2,
               dfloat* U,
               dfloat* AU) 
{
  int stride = mesh.Nx;
  int localNx = mesh.Nx-2;
  int localNy = mesh.Ny-2;
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
}
