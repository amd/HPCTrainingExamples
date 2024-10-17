//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include "Jacobi.hpp"

dfloat Norm(mesh_t& mesh, dfloat *U) 
{
  dfloat norm = 0.0;
  const int N = mesh.N;
  const dfloat dx = mesh.dx;
  const dfloat dy = mesh.dy;
  for (int id=0; id < N; id++) {
    norm += U[id] * U[id] * dx * dy;
  }
  return sqrt(norm)/N;
}
