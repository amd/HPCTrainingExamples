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
