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
#include <omp.h>
#include <cmath>
#include <cstddef>

void OpenMPExecutor::compute(std::vector<double>& data) {
    // For each element, perform 100 iterations of a heavy computation.
    #pragma omp parallel for
    for (std::size_t i = 0; i < data.size(); ++i) {
        double val = data[i];
        for (int j = 0; j < 100; ++j) {
            // A compute-bound operation: repeatedly update val.
            val = std::sin(val) + std::cos(val) + std::sqrt(std::fabs(val) + 1.0);
        }
        data[i] = val;
    }
}
