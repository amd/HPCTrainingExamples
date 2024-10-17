//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include "Jacobi.hpp"

#pragma omp requires unified_shared_memory
//Jacobi iterative method
// U = U + D^{-1}*(RHS - AU)
void Update(mesh_t& mesh,
            const dfloat factor,
            dfloat* RHS,
            dfloat* AU,
            dfloat* RES,
            dfloat* U) 
{
  const int N = mesh.N;
  #pragma omp target teams distribute parallel for
  for (int id=0;id<N;id++) 
  {
    const dfloat r_res = RHS[id] - AU[id];
    RES[id] = r_res;
    U[id] += r_res*factor;
  }
}
