#ifndef ASYNCH_JACOBI_PRECONDITIONER_H
#define ASYNCH_JACOBI_PRECONDITIONER_H
#define SUBWF_SIZE 32 // size of sub-wavefront
#include "preconditioner.h"

int setup_asynch_jacobi_preconditioner(const CSRMatrix& A,
                                       PreconditionerData& precond_data,
                                       rocsparse_handle handle_rocsparse);

int apply_asynch_jacobi_preconditioner(double* d_x,
                                       const double* d_r,
                                       const CSRMatrix& A,
                                       PreconditionerData& precond_data);

void cleanup_asynch_jacobi_preconditioner(PreconditionerData& precond_data);

#endif
