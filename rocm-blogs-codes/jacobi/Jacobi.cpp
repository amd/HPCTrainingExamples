//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include <unistd.h>
#include <string>
#include <chrono>
#include "Jacobi.hpp"

#ifdef _HIP
#define HD_U d_U
#define HD_AU d_AU
#define HD_RHS d_RHS
#define HD_RES d_RES
#else
#define HD_U h_U
#define HD_AU h_AU
#define HD_RHS h_RHS
#define HD_RES h_RES
#endif

Jacobi_t::Jacobi_t(mesh_t& mesh_):
  mesh(mesh_)
{
  CreateMesh();
  InitializeData();
}


Jacobi_t::~Jacobi_t() {
  SafeHostFree(mesh.x);
  SafeHostFree(mesh.y);
  SafeHostFree(h_U);
  SafeHostFree(h_AU);
  SafeHostFree(h_RHS);
  SafeHostFree(h_RES);
}

// Generates the 2D mesh
void Jacobi_t::CreateMesh() 
{

  mesh.N = mesh.Nx * mesh.Ny;

  //domain dimensions
  mesh.Lx = (X_MAX) - (X_MIN);
  mesh.Ly = (Y_MAX) - (Y_MIN);

  //mesh spacing
  mesh.dx = mesh.Lx/(mesh.Nx+1);
  mesh.dy = mesh.Ly/(mesh.Ny+1);

  uint64 Ntotal = mesh.N;
  mesh.invNtotal = 1.0/Ntotal;

  //coordinates (including boundary points)
  mesh.x = (dfloat *) malloc((mesh.Nx+2)*sizeof(dfloat));
  mesh.y = (dfloat *) malloc((mesh.Ny+2)*sizeof(dfloat));

  for (int i=0;i<mesh.Nx;i++)
    mesh.x[i] = mesh.dx + i*mesh.dx;

  for (int j=0;j<mesh.Ny;j++)
    mesh.y[j] = mesh.dy + j*mesh.dy;
}

// This allocates and initializes all the relevant data buffers before the Jacobi run
void Jacobi_t::InitializeData() 
{

  //host buffers
  h_U   = (dfloat*) malloc(mesh.N*sizeof(dfloat));
  h_AU  = (dfloat*) malloc(mesh.N*sizeof(dfloat));
  h_RHS = (dfloat*) malloc(mesh.N*sizeof(dfloat));
  h_RES = (dfloat*) malloc(mesh.N*sizeof(dfloat));
#ifdef _HIP
  hipMalloc(&d_U, mesh.N*sizeof(dfloat));
  hipMalloc(&d_AU, mesh.N*sizeof(dfloat));
  hipMalloc(&d_RES, mesh.N*sizeof(dfloat));
  hipMalloc(&d_RHS, mesh.N*sizeof(dfloat));
  NormSumMalloc(mesh);
  // Kernel launch configurations
  mesh.block.x = BLOCK_SIZE;
  mesh.grid.x = std::ceil(static_cast<double>(mesh.Nx * mesh.Ny) / mesh.block.x);
  mesh.grid2.x = std::ceil((2.0*mesh.Nx+2.0*mesh.Ny-2.0)/mesh.block.x);
  // Note: Kernel launch configurations for NormKernel in Norm.cpp
#endif

  #pragma omp parallel for collapse(2)
  for (int j=0;j<mesh.Ny;j++) 
  {
    for (int i=0;i<mesh.Nx;i++) 
    {
      int id = i+j*mesh.Nx;
      h_U[id] = 0.0; //initial guess
      h_RHS[id] = 0.0; //forcing
    }
  }

  // add boundary contributions
  uint64_t offsetX = 1;
  uint64_t offsetY = 1;
  dfloat totalX  = mesh.Nx+2;
  dfloat totalY  = mesh.Ny+2;

  // bottom side
  {
    for (int i=0;i<mesh.Nx;i++) 
    {
      dfloat bc = sin(M_PI*(i+offsetX)/totalX);
      h_RHS[i] += bc/(mesh.dx*mesh.dx);
    }
  }
  // top side
  {
    int j = mesh.Ny -1;
    for (int i=0;i<mesh.Nx;i++) 
    {
      dfloat bc = sin(M_PI*(i+offsetX)/totalX);
      h_RHS[i+j*mesh.Nx] += bc/(mesh.dx*mesh.dx);
    }
  }
  // left side
  {
    for (int j=0;j<mesh.Ny;j++) 
    {
      dfloat bc = sin(M_PI*(j+offsetY)/(totalY));
      h_RHS[j*mesh.Nx] += bc/(mesh.dy*mesh.dy);
      
    }
  }

  // right side
  {
    for (int j=0;j<mesh.Ny;j++) 
    {
      dfloat bc = sin(M_PI*(j+offsetY)/(totalY));
      h_RHS[(mesh.Nx-1)+j*mesh.Nx] += bc/(mesh.dy*mesh.dy);
    }
  }

  #pragma omp parallel for collapse(2)
  for (int j=0;j<mesh.Ny;j++) 
  {
    for (int i=0;i<mesh.Nx;i++) 
    {
      int id = i+j*mesh.Nx;
      h_AU[id] = 0.0;
      h_RES[id] = h_RHS[id];
    }
  }
}

// Display a number in a pretty format
char * FormatNumber(double value, const char * suffix, char * printBuf) 
{
  std::string magnitude = " kMGT";
  size_t orderIdx = 0;

  value = fabs(value);
  while ((value > 1000.0) && (orderIdx < magnitude.length() - 1))
  {
    ++orderIdx;
    value /= 1000.0;
  }

  sprintf(printBuf, "%.2lf %c%s", value, magnitude[orderIdx], suffix);

  return printBuf;
}

// Print a performance counter in a specific format
void PrintPerfCounter(const char * counterDesc, const char * counterUnit,
                      double counter, double elapsedTime, int size) {
  char printBuf[256];
  double avgCounter = counter / elapsedTime;

  printf("%s: %s \n", counterDesc, FormatNumber(avgCounter, counterUnit, printBuf));
}

void Jacobi_t::PrintResults() {
  double lattUpdates = 0.0, flops = 0.0, bandWidth = 0.0;

  // Show the performance counters
  printf("Total Jacobi run time: %.4lf sec.\n", elasped);

  // Compute the performance counters
  lattUpdates = 1.0 * (mesh.Nx) * (mesh.Ny) * iterations;
  flops = 17.0 * (lattUpdates);              // Operations per Jacobi kernel run
  bandWidth = 12.0 * (lattUpdates) * sizeof(dfloat);     // Transfers per Jacobi kernel run

  int size = 1;
  PrintPerfCounter("Measured lattice updates", "LU/s", lattUpdates, elasped, size);
  PrintPerfCounter("Measured FLOPS", "FLOPS", flops, elasped, size);
  PrintPerfCounter("Measured device bandwidth", "B/s", bandWidth, elasped, size);
  printf("Measured AI=%f\n", flops/bandWidth);
}

void Jacobi_t::Run() 
{
  const int N = mesh.N;
#ifdef _HIP
  hipMemcpy(d_U, h_U, N*sizeof(dfloat), hipMemcpyHostToDevice);
  hipMemcpy(d_AU, h_AU, N*sizeof(dfloat), hipMemcpyHostToDevice);
  hipMemcpy(d_RES, h_RES, N*sizeof(dfloat), hipMemcpyHostToDevice);
  hipMemcpy(d_RHS, h_RHS, N*sizeof(dfloat), hipMemcpyHostToDevice);
#else
  #pragma omp target enter data map(to: mesh,h_U[0:N],h_AU[0:N],h_RES[0:N],h_RHS[0:N])
#endif

  std::cout << "Starting Jacobi run" << std::endl;
  iterations = 0;

  //compute initial residual (assumes zero initial guess)
#ifdef _HIP
  dfloat residual = Norm(mesh, d_RES);
#else
  dfloat residual = Norm(mesh, h_RES);
#endif
  std::cout << "Iteration:" << iterations << " - Residual:" << residual << std::endl;
  
  // Scalar factor used in Jacobi method
  const dfloat factor = 1/(2.0/(mesh.dx*mesh.dx) + 2.0/(mesh.dy*mesh.dy));
  const dfloat _1bydx2 = 1.0/(mesh.dx*mesh.dx);
  const dfloat _1bydy2 = 1.0/(mesh.dy*mesh.dy);

  auto timerStart = std::chrono::high_resolution_clock::now();

  while ((iterations < JACOBI_MAX_LOOPS) && (residual > JACOBI_TOLERANCE)) 
  {
    // Compute Laplacian
    Laplacian(mesh, _1bydx2, _1bydy2, HD_U, HD_AU);

    // Apply Boundary Conditions
    BoundaryConditions(mesh, _1bydx2, _1bydy2, HD_U, HD_AU);

    // Update the solution
    Update(mesh, factor, HD_RHS, HD_AU, HD_RES, HD_U);

    // Compute residual = ||U||
    residual = Norm(mesh, HD_RES);
    
    ++iterations;
    if(iterations % 100 == 0)
      std::cout << "Iteration:" << iterations << " - Residual:" << residual << std::endl;
  }

  auto timerStop = std::chrono::high_resolution_clock::now();
  // time in secs = nanosecs * 1.e-9
  elasped = std::chrono::duration_cast<std::chrono::nanoseconds>(timerStop - timerStart).count() * 1.e-9;

  std::cout << "Stopped after " << iterations << " with residue:" << residual << std::endl;
  
  PrintResults();
#ifdef _HIP
  hipFree(d_U);
  hipFree(d_AU);
  hipFree(d_RHS);
  hipFree(d_RES);
  hipHostFree(mesh.norm_sum);
#else
  #pragma omp target exit data map(release: h_U[0:N],h_AU[0:N],h_RES[0:N],h_RHS[0:N])
#endif
}
