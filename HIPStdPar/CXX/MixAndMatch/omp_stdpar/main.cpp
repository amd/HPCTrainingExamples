#include "OpenMPExecutor.hpp"
#include "StdParExecutor.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <chrono>

using namespace std::chrono;

void printFirstN(const std::vector<double>& data, std::size_t n = 10) {
    for (std::size_t i = 0; i < n && i < data.size(); ++i)
        std::cout << data[i] << " ";
    std::cout << std::endl;
}

int main() {
    const std::size_t numElements = 100000000;
    std::vector<double> data(numElements);

    // Initialize random number generator.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 100.0);

    // Fill the vector with random values.
    std::generate(data.begin(), data.end(), [&]() { return dis(gen); });

    // Process with OpenMP executor.
    {
        std::vector<double> copy = data;
        std::unique_ptr<ParallelExecutor> executor = std::make_unique<OpenMPExecutor>();

        auto start = high_resolution_clock::now();
        executor->compute(copy);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start).count();

        std::cout << "OpenMP Executor first 10 results: ";
        printFirstN(copy);
        std::cout << "OpenMP Executor took " << duration << " ms" << std::endl;
    }

    // Process with std::execution::par_unseq executor.
    {
        std::vector<double> copy = data;
        std::unique_ptr<ParallelExecutor> executor = std::make_unique<StdParExecutor>();

        auto start = high_resolution_clock::now();
        executor->compute(copy);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start).count();

        std::cout << "StdPar Executor first 10 results: ";
        printFirstN(copy);
        std::cout << "StdPar Executor took " << duration << " ms" << std::endl;
    }

    return 0;
}
