#ifndef PARALLEL_EXECUTOR_HPP
#define PARALLEL_EXECUTOR_HPP

#include <vector>

class ParallelExecutor {
public:
    virtual ~ParallelExecutor() = default;
    // Process the vector and, for demonstration, accumulate a computed value.
    virtual void compute(std::vector<double>& data) = 0;
};

#endif // PARALLEL_EXECUTOR_HPP
