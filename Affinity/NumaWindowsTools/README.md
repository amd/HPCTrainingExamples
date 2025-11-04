# Numa Windows Tools

**NumaWindowsTools** is a small Windows library that provides functions to retrieve the **NUMA node** of your GPUs.  
On Linux, you can use hwloc: https://www.open-mpi.org/projects/hwloc/

The project includes a simple demo (`main.cpp`) showing how to use the library.  
It uses HIP to enumerate GPUs and NumaWindowsTools to query the NUMA node for each GPU.

> Your system must be configured with multiple NUMA nodes; otherwise, calls to `NumaWindowsTool::GetNumaNodeForPciBdf` will return invalid results.

---

## Build Instructions of the Demo

You can install ROCm on Windows following [these](https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html) instructions.
The installation path will be `C:/Program Files/AMD/ROCm/x.x`.

```bat
cmake -B build -G "Visual Studio 17 2022" -DCMAKE_PREFIX_PATH="C:/Program Files/AMD/ROCm/6.4"
cmake --build build --config Release
```
Make sure to use the correct HIP SDK path ( https://www.amd.com/en/developer/resources/rocm-hub/hip-sdk.html )

