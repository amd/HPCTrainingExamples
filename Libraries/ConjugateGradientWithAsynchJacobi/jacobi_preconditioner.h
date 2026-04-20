#ifndef JACOBI_PRECONDITIONER_H
#define JACOBI_PRECONDITIONER_H

#include "preconditioner.h"

int setup_jacobi_preconditioner(const CSRMatrix& A,
                                PreconditionerData& precond_data,
                                rocsparse_handle handle_rocsparse);

int apply_jacobi_preconditioner(double* d_x,
                                const double* d_r,
                                const CSRMatrix& A,
                                PreconditionerData& precond_data);

void cleanup_jacobi_preconditioner(PreconditionerData& precond_data);

#endif
