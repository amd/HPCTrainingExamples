//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#pragma once
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

#ifdef _HIP
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256

#define GET_LOCAL_ID_0 (__builtin_amdgcn_workitem_id_x())
#define GET_LOCAL_ID_1 (__builtin_amdgcn_workitem_id_y())
#define GET_LOCAL_ID_2 (__builtin_amdgcn_workitem_id_z())

#define GET_BLOCK_ID_0 (__builtin_amdgcn_workgroup_id_x())
#define GET_BLOCK_ID_1 (__builtin_amdgcn_workgroup_id_y())
#define GET_BLOCK_ID_2 (__builtin_amdgcn_workgroup_id_z())

#define GET_BLOCK_SIZE_0 (__builtin_amdgcn_workgroup_size_x())
#define GET_BLOCK_SIZE_1 (__builtin_amdgcn_workgroup_size_y())
#define GET_BLOCK_SIZE_2 (__builtin_amdgcn_workgroup_size_z())

#define GET_GLOBAL_ID_0 (GET_LOCAL_ID_0 + GET_BLOCK_SIZE_0 * GET_BLOCK_ID_0)
#define GET_GLOBAL_ID_1 (GET_LOCAL_ID_1 + GET_BLOCK_SIZE_1 * GET_BLOCK_ID_1)
#define GET_GLOBAL_ID_2 (GET_LOCAL_ID_2 + GET_BLOCK_SIZE_2 * GET_BLOCK_ID_2)


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(hipError_t code, const char *file, int line, bool abort=true)
{
   if (code != hipSuccess) 
   {
      fprintf(stdout,"GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
      if (abort) std::exit(code);
   }
}
#endif


// Setting this to 1 makes the application use only single-precision floating-point data. Set this to
// 0 in order to use double-precision floating-point data instead.
#define USE_FLOAT			0

// This is the default domain size (when not explicitly stated with "-d" in the command-line arguments).
#define DEFAULT_DOMAIN_SIZE 4096

// This is the minimum acceptable domain size in any of the 2 dimensions.
#define MIN_DOM_SIZE		1

// Global domain dimensions
#define X_MIN -0.5
#define X_MAX  0.5
#define Y_MIN -0.5
#define Y_MAX  0.5

// Side Indices
#define NSIDES     4
#define SIDE_DOWN  0
#define SIDE_RIGHT 1
#define SIDE_UP    2
#define SIDE_LEFT  3

// This is the Jacobi tolerance threshold. The run is considered to have converged when the maximum residue
// falls below this value.
#define	JACOBI_TOLERANCE	1.0E-5F

// This is the Jacobi iteration count limit. The Jacobi run will never cycle more than this, even if it
// has not converged when finishing the last allowed iteration.
#define JACOBI_MAX_LOOPS	1000

// This is the status value that indicates a successful operation.
#define STATUS_OK 			0

// This is the status value that indicates an error.
#define STATUS_ERR			-1

#if USE_FLOAT
	#define dfloat				float
	#define MPI_DFLOAT		MPI_FLOAT
#else
	#define dfloat				double
	#define MPI_DFLOAT		MPI_DOUBLE
#endif

#define uint64					unsigned long long
#define MPI_UINT64			MPI_UNSIGNED_LONG_LONG

#define SafeHostFree(block)			{ if (block) free(block); }
#define OnePrintf(allow, ...)		{ if (allow) printf(__VA_ARGS__); }
#define OneErrPrintf(allow, ...)	{ if (allow) fprintf(stderr, __VA_ARGS__); }

// This contains mesh information
struct mesh_t 
{
  int N;
  int Nx, Ny;

  dfloat Lx, Ly;
  dfloat dx, dy;

  dfloat invNtotal;

  dfloat *x;
  dfloat *y;

#ifdef _HIP
  dim3 block;
  dim3 grid,grid2;
  // Norm buffer
  dfloat *norm_sum;
#endif
};

class Jacobi_t {
private:
  mesh_t& mesh;

  //host buffers
  dfloat *h_U, *h_RHS, *h_AU, *h_RES;
#ifdef _HIP
  //device buffers
  dfloat *d_U, *d_AU, *d_RES, *d_RHS;
#endif

  double timerStart, timerStop, elasped, avgTransferTime;
  double totalCommTime, totalLocalComputeTime;
  int iterations;

  void ApplyTopology();

  void CreateMesh();

  void InitializeData();

  void PrintResults();

public:
  Jacobi_t(mesh_t& mesh_);

  ~Jacobi_t();

  void Run();
};

dfloat ForcingFunction(dfloat x, dfloat y);
dfloat BoundaryFunction(dfloat x, dfloat y);

void ParseCommandLineArguments(int argc, char ** argv, mesh_t& mesh);

void NormSumMalloc(mesh_t& mesh);

dfloat Norm(mesh_t& mesh, dfloat *U);

void Laplacian(mesh_t& mesh,
               const dfloat _1bydx2,
               const dfloat _1bydy2,
               dfloat* d_U,
               dfloat* d_AU);

void BoundaryConditions(mesh_t& mesh,
                        const dfloat _1bydx2,
                        const dfloat _1bydy2,
                        dfloat* d_U,
                        dfloat* d_AU);

void Update(mesh_t& mesh,
            const dfloat factor,
            dfloat* d_RHS,
            dfloat* d_AU,
            dfloat* d_RES,
            dfloat* d_U);

