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
