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

#include "OpenMPExecutor.hpp"
#include "StdParExecutor.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std::chrono;

int main() {
    const std::size_t numElements = 10000;
    std::vector<double> data(numElements);

    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 100.0);

    // Fill the vector with random values.
    std::generate(data.begin(), data.end(), [&]() { return dis(gen); });

    // Process with OpenMP executor.
    {
        std::unique_ptr<ParallelExecutor> executor = std::make_unique<OpenMPExecutor>();
        auto start = high_resolution_clock::now();
        executor->compute(data);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        std::cout << "OpenMP computation took " << duration << " ms\n";
    }

    // Process with std::execution::par_unseq executor.
    {
        std::unique_ptr<ParallelExecutor> executor = std::make_unique<StdParExecutor>();
        auto start = high_resolution_clock::now();
        executor->compute(data);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start).count();
        std::cout << "StdPar computation took " << duration << " ms\n";
    }

    return 0;
}
