/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <vector>
#include <algorithm>
#include <execution>
#include <random>
#include <chrono>

// HIP headers
#include <hip/hip_runtime.h>

using namespace std::chrono;

// -----------------------------------------------------------------------------
// Option 1: Explicit device allocation + copy + stdpar offload + copy-back
// -----------------------------------------------------------------------------
void runExplicitDataMovement(std::vector<float>& hostData)
{
    const size_t N = hostData.size();

    // Allocate device memory
    float* d_data = nullptr;
    hipError_t err = hipMalloc(&d_data, N * sizeof(float));
    if (err != hipSuccess) {
        std::cerr << "Error allocating GPU memory: " << hipGetErrorString(err) << "\n";
        return;
    }

    // Copy host -> device
    err = hipMemcpy(d_data, hostData.data(), N * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        std::cerr << "Error copying data to GPU: " << hipGetErrorString(err) << "\n";
        hipFree(d_data);
        return;
    }

    // Time just the stdpar offload (NOT including the device->host copy).
    auto startOffload = high_resolution_clock::now();

    // Offload the for_each with par_unseq, operating directly on device memory
    std::for_each(std::execution::par_unseq, d_data, d_data + N, [] __device__ (float &x) {
        // For demonstration, square each element
        x = x * x;
    });

    auto endOffload = high_resolution_clock::now();
    auto offloadDuration = duration_cast<milliseconds>(endOffload - startOffload).count();

    // Copy device -> host
    auto startCopyBack = high_resolution_clock::now();
    err = hipMemcpy(hostData.data(), d_data, N * sizeof(float), hipMemcpyDeviceToHost);
    auto endCopyBack = high_resolution_clock::now();
    auto copyBackDuration = duration_cast<milliseconds>(endCopyBack - startCopyBack).count();

    if (err != hipSuccess) {
        std::cerr << "Error copying data back to host: " << hipGetErrorString(err) << "\n";
        hipFree(d_data);
        return;
    }

    hipFree(d_data);

    std::cout << "=== Explicit Data Movement Timing ===\n"
              << "Offload time (GPU kernel only): " << offloadDuration << " ms\n"
              << "Copy-back time (device->host):  " << copyBackDuration << " ms\n\n";
}


// -----------------------------------------------------------------------------
// Option 2: "No Explicit Data Movement"
// This simulates a scenario where your compiler/runtime might automatically
// move data behind the scenes (e.g., unified memory or implicit transfers).
// -----------------------------------------------------------------------------
void runNoExplicitDataMovement(std::vector<float>& hostData)
{
    const size_t N = hostData.size();

    // In a true unified-memory scenario, you'd allocate your data differently,
    // e.g., using hipMallocManaged(). For simplicity, let's just do it on the host
    // and assume the compiler can offload automatically. This is highly compiler-
// dependent. Some toolchains require special flags or won't do it at all.

    auto startOffload = high_resolution_clock::now();

    // Potential offload if supported by the toolchain.
    // If not supported, this might just run on the CPU (with vectorization/threads).
    std::for_each(std::execution::par_unseq, hostData.begin(), hostData.end(), [] (float &x) {
        // Square each element
        x = x * x;
    });

    auto endOffload = high_resolution_clock::now();
    auto offloadDuration = duration_cast<milliseconds>(endOffload - startOffload).count();

    std::cout << "=== No Explicit Data Movement Timing ===\n"
              << "Offload time (potential GPU or CPU fallback): " << offloadDuration << " ms\n\n";
}


// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main()
{
    const size_t N = 500000000;
    std::vector<float> data(N);

    // Initialize random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 100.0f);
    for (auto &val : data) {
        val = dis(gen);
    }

    // We'll create two copies of this data so that
    // each run starts from the same initial values.
    auto dataCopyForExplicit = data; 
    auto dataCopyForNoMove  = data;

    // --------------------------------------------------
    // 1) EXPLICIT DATA MOVEMENT
    // --------------------------------------------------
    runExplicitDataMovement(dataCopyForExplicit);

    // Show a few results
    std::cout << "First 5 results (explicit copy): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << dataCopyForExplicit[i] << " ";
    }
    std::cout << "\n\n";

    // --------------------------------------------------
    // 2) "NO EXPLICIT DATA MOVEMENT"
    // --------------------------------------------------
    runNoExplicitDataMovement(dataCopyForNoMove);

    // Show a few results
    std::cout << "First 5 results (no explicit copy): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << dataCopyForNoMove[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
