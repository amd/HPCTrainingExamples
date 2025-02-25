#include "StdParCpuExecutor.hpp"
#include <cmath>
#include <cstddef>
#include <execution>

void StdParCpuExecutor::compute(std::vector<double>& data) {
    std::transform(std::execution::par,
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
