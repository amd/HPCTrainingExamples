#include "StdParGpuExecutor.hpp"
#include <algorithm>
#include <execution>
#include <cmath>

void StdParGpuExecutor::compute(std::vector<double>& data) {
    // Use std::transform with the parallel unsequenced execution policy.
    std::transform(std::execution::par_unseq,
                   data.begin(), data.end(),
                   data.begin(),
                   [](double x) {
                       // Perform 100 iterations of a heavy computation.
                       for (int j = 0; j < 100; ++j) {
                           x = std::sin(x) + std::cos(x) + std::sqrt(std::fabs(x) + 1.0);
                       }
                       return x;
                   });
}
