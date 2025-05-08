//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#pragma once
#include <iostream>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>



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

};

class Jacobi_t {
private:
  mesh_t& mesh;

  //host buffers
  dfloat *h_U, *h_RHS, *h_AU, *h_RES;

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

