#include "preconditioner.h"
#include "ic_preconditioner.h"
#include "jacobi_preconditioner.h"
#include "asynch_jacobi_preconditioner.h"
#include <cstdio>

int setup_preconditioner(const std::string& name,
                         const CSRMatrix& A,
                         PreconditionerData& precond_data,
                         rocsparse_handle handle_rocsparse)
{
  if (name == "ic" || name == "IC" || name == "ichol") {
    return setup_ic_preconditioner(A, precond_data, handle_rocsparse);
  } else if (name == "it_jacobi" || name == "jacobi") {
    return setup_jacobi_preconditioner(A, precond_data, handle_rocsparse);
  } else if (name == "asynch_it_jacobi" || name == "asynch_jacobi") {
    return setup_asynch_jacobi_preconditioner(A, precond_data, handle_rocsparse);
  } else if (name == "none" || name == "NONE") {
    precond_data.name = "none";
    precond_data.n = A.n;
    return 0;
  } else {
    printf("Unknown preconditioner: %s\n", name.c_str());
    return -1;
  }
}

int apply_preconditioner(const std::string& name,
                         double* d_x,
                         const double* d_r,
                         const CSRMatrix& A,
                         PreconditionerData& precond_data)
{
  if (name == "ic" || name == "IC" || name == "ichol") {
    return apply_ic_preconditioner(d_x, d_r, A, precond_data);
  } else if (name == "it_jacobi" || name == "jacobi") {
    return apply_jacobi_preconditioner(d_x, d_r, A, precond_data);
  } else if (name == "asynch_it_jacobi" || name == "asynch_jacobi") {
    return apply_asynch_jacobi_preconditioner(d_x, d_r, A, precond_data);
  } else if (name == "none" || name == "NONE") {
    HIP_CHECK(hipMemcpy(d_x, d_r, sizeof(double) * precond_data.n, hipMemcpyDeviceToDevice));
    return 0;
  } else {
    printf("Unknown preconditioner: %s\n", name.c_str());
    return -1;
  }
}

void cleanup_preconditioner(PreconditionerData& precond_data)
{
  if (precond_data.name == "ic" || precond_data.name == "IC" || precond_data.name == "ichol") {
    cleanup_ic_preconditioner(precond_data);
  } else if (precond_data.name == "it_jacobi" || precond_data.name == "jacobi") {
    cleanup_jacobi_preconditioner(precond_data);
  } else if (precond_data.name == "asynch_it_jacobi" || precond_data.name == "asynch_jacobi") {
    cleanup_asynch_jacobi_preconditioner(precond_data);
  }
}
