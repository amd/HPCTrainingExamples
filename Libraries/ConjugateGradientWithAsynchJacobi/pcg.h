#ifndef PCG_H
#define PCG_H

#include "hip_utils.h"
#include "preconditioner.h"

struct PCGResult {
  int iterations;
  double final_residual;
  int flag;  // 0 = converged, 1 = maxit reached, -1 = failed
};

PCGResult pcg_solve(const CSRMatrix& A,
                    double* d_x,
                    const double* d_b,
                    int maxit,
                    double tol,
                    const std::string& precond_name,
                    PreconditionerData& precond_data,
                    rocblas_handle handle_rocblas,
                    rocsparse_handle handle_rocsparse);

#endif
