#ifndef ITERATIVE_GAUSS_SEIDEL_H
#define ITERATIVE_GAUSS_SEIDEL_H

#include "preconditioner.h"

int setup_gs_preconditioner(const CSRMatrix& A,
                            PreconditionerData& precond_data,
                            rocsparse_handle handle_rocsparse);

int apply_gs_it_preconditioner(double* d_x,
                               const double* d_b,
                               const CSRMatrix& A,
                               PreconditionerData& precond_data);

int apply_gs_it2_preconditioner(double* d_x,
                                const double* d_b,
                                const CSRMatrix& A,
                                PreconditionerData& precond_data);

void cleanup_gs_preconditioner(PreconditionerData& precond_data);

#endif
