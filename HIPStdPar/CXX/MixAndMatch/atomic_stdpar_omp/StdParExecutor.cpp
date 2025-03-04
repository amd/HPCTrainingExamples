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

#include "StdParExecutor.hpp"
#include <algorithm>
#include <execution>
#include <cmath>
#include <atomic>
#include <iostream>

void StdParExecutor::compute(std::vector<double>& data) {
    // Use std::atomic for a lock-free (if available) accumulation.
    std::atomic<double> sum{0.0};
    //double sum = 0.0;
    // Note: std::atomic<double> may not be lock-free on all platforms.
    
    // Parallel unsequenced execution with std::for_each.
    std::for_each(std::execution::par_unseq, data.begin(), data.end(),
        [&](double x) {
            // Heavy computation: 100 iterations.
            for (int j = 0; j < 100; ++j) {
                x = std::sin(x) + std::cos(x) + std::sqrt(std::fabs(x) + 1.0);
            }
            // Atomically add the computed value.
            sum.fetch_add(x,std::memory_order_relaxed);
	    //__atomic_fetch_add(&sum, x, __ATOMIC_SEQ_CST);
        });
    std::cout << "StdPar Accumulated Sum: " << sum.load() << std::endl;
}
