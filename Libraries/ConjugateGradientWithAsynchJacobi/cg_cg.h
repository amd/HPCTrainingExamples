#ifndef CG_CG_H
#define CG_CG_H

#include "hip_utils.h"
#include "preconditioner.h"

struct CGCGResult {
  int iterations;
  double final_residual;
  int flag;  // 0 = converged, 1 = maxit reached, -1 = failed
};

CGCGResult cg_cg_solve(const CSRMatrix& A,
                       double* d_x,
                       const double* d_b,
                       int maxit,
                       double tol,
                       const std::string& precond_name,
                       PreconditionerData& precond_data,
                       rocblas_handle handle_rocblas,
                       rocsparse_handle handle_rocsparse,
                       int low_synch);

#endif
