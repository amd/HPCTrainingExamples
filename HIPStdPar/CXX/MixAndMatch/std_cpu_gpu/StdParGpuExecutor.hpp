#ifndef STDPAR_EXECUTOR_HPP
#define STDPAR_EXECUTOR_HPP

#include "ParallelExecutor.hpp"
#include <vector>

class StdParGpuExecutor : public ParallelExecutor {
public:
    void compute(std::vector<double>& data) override;
};

#endif // STDPAR_EXECUTOR_HPP
