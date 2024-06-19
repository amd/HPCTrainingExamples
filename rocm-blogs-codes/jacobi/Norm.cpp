//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include "Jacobi.hpp"

#ifdef _HIP
#define NORM_BLOCK_SIZE 512
#define NORM_NUM_BLOCKS 256
__global__
__launch_bounds__(NORM_BLOCK_SIZE)
void NormKernel(const dfloat * a, dfloat * sum, int N)
{
  __shared__ dfloat smem[NORM_BLOCK_SIZE];

  int i = GET_GLOBAL_ID_0;
  const size_t si = GET_LOCAL_ID_0;

  smem[si] = 0.0;
  for (; i < N; i += NORM_BLOCK_SIZE * NORM_NUM_BLOCKS)
    smem[si] += a[i] * a[i];

  for (int offset = NORM_BLOCK_SIZE >> 1; offset > 0; offset >>= 1)
  {
    __syncthreads();
    if (si < offset) smem[si] += smem[si+offset];
  }

  if (si == 0)
    sum[GET_BLOCK_ID_0] = smem[si];
}

void NormSumMalloc(mesh_t &mesh)
{
  hipHostMalloc(&mesh.norm_sum, NORM_NUM_BLOCKS*sizeof(dfloat), hipHostMallocNonCoherent);
}

#endif

dfloat Norm(mesh_t& mesh, dfloat *U) 
{
  dfloat norm = 0.0;
  const int N = mesh.N;
  const dfloat dx = mesh.dx;
  const dfloat dy = mesh.dy;
#if _HIP
  NormKernel<<<NORM_NUM_BLOCKS,NORM_BLOCK_SIZE>>>(U, mesh.norm_sum, N);
  hipDeviceSynchronize();
  for (int id=0; id < NORM_NUM_BLOCKS; id++)
    norm += mesh.norm_sum[id];
  return sqrt(norm*dx*dy)*mesh.invNtotal;
#else
  #pragma omp target teams distribute parallel for reduction(+:norm)
  for (int id=0; id < N; id++) {
    norm += U[id] * U[id] * dx * dy;
  }
  return sqrt(norm)/N;
#endif
}
