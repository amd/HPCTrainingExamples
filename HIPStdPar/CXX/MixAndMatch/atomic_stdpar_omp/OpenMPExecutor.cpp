#include "OpenMPExecutor.hpp"
#include <omp.h>
#include <cmath>
#include <cstddef>
#include <iostream>

void OpenMPExecutor::compute(std::vector<double>& data) {
    double sum = 0.0;

    // Parallel loop with OpenMP.
    #pragma omp parallel for
    for (std::size_t i = 0; i < data.size(); ++i) {
        double tmp = data[i];

        // Heavy, compute-bound workload: perform 100 iterations.
        for (int j = 0; j < 100; ++j) {
            tmp = std::sin(tmp) + std::cos(tmp) + std::sqrt(std::fabs(tmp) + 1.0);
        }

        // Safely accumulate using the OpenMP atomic directive.
        #pragma omp atomic
        sum += tmp;
    }
    std::cout << "OpenMP Accumulated Sum: " << sum << std::endl;
}
