//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include "Jacobi.hpp"

//Jacobi iterative method
// U = U + D^{-1}*(RHS - AU)
#ifdef _HIP
__global__
__launch_bounds__(BLOCK_SIZE)
void UpdateKernel(const int N,
                  const dfloat factor,
                  const dfloat * RHS,
                  const dfloat * AU,
                  dfloat * RES,
                  dfloat * U)
{
  int tid = GET_GLOBAL_ID_0;
  dfloat r_res;
  for (int i = tid; i < N; i += gridDim.x * blockDim.x)
  {
    r_res = RHS[tid] - AU[tid];
    RES[tid] = r_res;
    U[tid] += r_res*factor;
  }
}
#endif
void Update(mesh_t& mesh,
            const dfloat factor,
            dfloat* RHS,
            dfloat* AU,
            dfloat* RES,
            dfloat* U) 
{
#ifdef _HIP
  UpdateKernel<<<mesh.grid,mesh.block>>>(mesh.N, factor, RHS, AU, RES, U);
#else
  const int N = mesh.N;
  #pragma omp target teams distribute parallel for
  for (int id=0;id<N;id++) 
  {
    const dfloat r_res = RHS[id] - AU[id];
    RES[id] = r_res;
    U[id] += r_res*factor;
  }
#endif
}
