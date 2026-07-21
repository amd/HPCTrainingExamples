#ifndef IC_PRECONDITIONER_H
#define IC_PRECONDITIONER_H

#include "preconditioner.h"

int setup_ic_preconditioner(const CSRMatrix& A,
                            PreconditionerData& precond_data,
                            rocsparse_handle handle_rocsparse);

int apply_ic_preconditioner(double* d_x,
                            const double* d_r,
                            const CSRMatrix& A,
                            PreconditionerData& precond_data);

void cleanup_ic_preconditioner(PreconditionerData& precond_data);

#endif
