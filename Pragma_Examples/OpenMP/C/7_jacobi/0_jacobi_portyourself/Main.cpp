//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include "Jacobi.hpp"

// This contains the application entry point
int main(int argc, char ** argv)
{
  mesh_t mesh;

  // Extract topology and domain dimensions from the command-line arguments
  ParseCommandLineArguments(argc, argv, mesh);

  Jacobi_t Jacobi(mesh);

  Jacobi.Run();

  return STATUS_OK;
}
