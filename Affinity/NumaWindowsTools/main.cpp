/*
MIT License
Copyright (c) 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

/*

This main.cpp is just a demo using the 'NumaWindowsTools' lib

It's using HIP to list the GPU, and 'NumaWindowsTools' to get the NUMA node of each GPU.

You machine must be configured in a multi-NUMA node in order to have valide calls to NumaWindowsTool::GetNumaNodeForPciBdf.

*/


#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include "NumaWindowsTools.h"

static void checkHip(hipError_t e, const char* what)
{
	if (e != hipSuccess)
	{
		std::fprintf(stderr, "HIP error at %s: %s\n", what, hipGetErrorString(e));
		std::exit(1);
	}
}


int main()
{
	int count = 0;
	checkHip(hipGetDeviceCount(&count), "hipGetDeviceCount");


	int numeCount = NumaWindowsTool::CountSystemNUMAnodes();
	printf("Your system has %d NUMA nodes.\n\n" , numeCount );

	for (int i = 0; i < count; ++i)
	{
		hipDeviceProp_t prop = {};
		checkHip(hipGetDeviceProperties(&prop, i), "hipGetDeviceProperties");

		char bdf_str[64] = {0};
		hipError_t e_bdf = hipDeviceGetPCIBusId(bdf_str, static_cast<int>(sizeof(bdf_str)), i);
		if (e_bdf != hipSuccess)
		{
			printf("ERROR 'hipDeviceGetPCIBusId' for this GPU\n");
			continue;
		}

		unsigned int seg = 0, bus = 0, dev = 0, func = 0;
		int retparse = NumaWindowsTool::ParsePciBusId(bdf_str, &seg, &bus, &dev, &func);
		if ( retparse != 0 )
		{
			printf("ERROR 'ParsePciBusId' for this GPU\n");
			continue;
		}

		std::printf("Device %d: %s\n", i, prop.name);
		std::printf("  PCI BDF string : %s\n", bdf_str);
		std::printf("  Segment (domain): 0x%04x\n", seg);
		std::printf("  Bus              : 0x%02x\n", bus);
		std::printf("  Device           : 0x%02x\n", dev);
		std::printf("  Function         : 0x%x\n", func);

		if ( numeCount >= 2 )
		{
			int numaIndex = 0;
			int oknuma = NumaWindowsTool::GetNumaNodeForPciBdf(seg, bus, dev, func, &numaIndex );
			if ( oknuma != 0 )
			{
				printf("  ERROR 'GetNumaNodeForPciBdf' for this GPU\n");
			}
			else
			{
				printf("  NUMA index = %d\n", numaIndex );
			}
		}
		else
		{
			printf("  No NUMA config on this system.\n");
		}

		printf("\n\n");

	}

	return 0;
}


