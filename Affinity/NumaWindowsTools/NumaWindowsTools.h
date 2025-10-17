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


#ifndef NUMA_WINDOWS_TOOLS_LIB
#define NUMA_WINDOWS_TOOLS_LIB

namespace NumaWindowsTool
{
	// return the number of NUMA nodes in the system
	unsigned int CountSystemNUMAnodes();

	// get a NUMA index for a given PCI BDF device
	// This implementation is for Windows only.
	// Warning: if the system only has 1 NUMA node, this function will likely return fail because Windows API doesn't expose the NUMA getters in this case.
	//          so it's a good practice to call CountSystemNUMAnodes() first, and only call  GetNumaNodeForPciBdf is there is at least 2 NUMA nodes.
	// return 0 if success
	int GetNumaNodeForPciBdf(unsigned short seg_in, unsigned char  bus_in, unsigned char  dev_in, unsigned char  func_in, int *out_numa);

	// parse a PCIBusId string, example: "0000:2b:00.0" -> [0, 0x2b, 0, 0]
	// return 0 if success
	int ParsePciBusId(const char *s, unsigned *domain, unsigned *bus, unsigned *dev, unsigned *func);

};

#endif // #ifndef NUMA_WINDOWS_TOOLS_LIB
