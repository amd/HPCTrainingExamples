//**************************************************************************
//* Copyright (c) 2023, Advanced Micro Devices, Inc. All rights reserved.
//**************************************************************************

#include <string.h>
#include "Jacobi.hpp"

// This contains the command-line argument parser and support functions

// Print the usage information for this application
void PrintUsage(const char * appName)
{
	printf("Usage: %s [-m Mesh.X [Mesh.Y]] [-h | --help]\n", appName);
	printf(" -m Mesh.x [Mesh.y]: set the domain size per node (if \"Mesh.y\" is missing, the domain size will default to (Mesh.x, Mesh.x); Mesh.x and Mesh.y must be positive integers)\n");
	printf(" -h | --help: print help information\n");
}

// Find (and if found, erase) an argument in the command line
int FindAndClearArgument(const char * argName, int argc, char ** argv) {
	for(int i = 1; i < argc; ++i)	{
		if (strcmp(argv[i], argName) == 0) {
			strcpy(argv[i], "");
			return i;
		}
	}

	return -1;
}

// Extract a number given as a command-line argument
int ExtractNumber(int argIdx, int argc, char ** argv) {
	int result = 0;

	if (argIdx < argc) {
		result = atoi(argv[argIdx]);
		if (result > 0)	{
			strcpy(argv[argIdx], "");
		}
	}

	return result;
}

// Parses the application's command-line arguments
// - argc          The number of input arguments
// - argv          The input arguments
// - mesh          The mesh topology struct

void ParseCommandLineArguments(int argc, char ** argv, mesh_t& mesh) 
{
	int canPrint = 1;
	int argIdx;

	// If help is requested, all other arguments will be ignored
	if ((FindAndClearArgument("-h", argc, argv) != -1) || (FindAndClearArgument("--help", argc, argv) != -1))	{
		if (canPrint)
			PrintUsage(argv[0]);
		std::abort();
	}

	// The domain size information is optional
	argIdx = FindAndClearArgument("-m", argc, argv);
	if (argIdx == -1)	{
		mesh.Nx = mesh.Ny = DEFAULT_DOMAIN_SIZE;
	}
	else 
	{
		mesh.Nx = ExtractNumber(argIdx + 1, argc, argv);
		mesh.Ny = ExtractNumber(argIdx + 2, argc, argv);

		// At least the first domain dimension must be specified
		if (mesh.Nx < MIN_DOM_SIZE) 
		{
			OneErrPrintf(canPrint, "Error: The local domain size must be at least %d (currently: %d)\n", MIN_DOM_SIZE, mesh.Nx);
			std::abort();
		}

		// If the second domain dimension is missing, it will default to the first dimension's value
		if (mesh.Ny <= 0) 
		{
			mesh.Ny = mesh.Nx;
		}
	}

	// At the end, there should be no other arguments that haven't been parsed
	for (int i = 1; i < argc; ++i) 
	{
		if (strlen(argv[i]) > 0) 
		{
			OneErrPrintf(canPrint, "Error: Unknown argument (\"%s\")\n", argv[i]);
			std::abort();
		}
	}

	// If we reach this point, all arguments were parsed successfully
	if (canPrint)	
	{
		printf("Domain size : %d x %d\n", mesh.Nx, mesh.Ny);
	}
}

