#ifndef OPENMP_EXECUTOR_HPP
#define OPENMP_EXECUTOR_HPP

#include "ParallelExecutor.hpp"
#include <vector>

class StdParCpuExecutor : public ParallelExecutor {
public:
    void compute(std::vector<double>& data) override;
};

#endif // OPENMP_EXECUTOR_HPP
