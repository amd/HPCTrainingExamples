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
