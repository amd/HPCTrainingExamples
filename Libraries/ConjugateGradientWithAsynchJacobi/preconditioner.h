#ifndef PRECONDITIONER_H
#define PRECONDITIONER_H

#include "hip_utils.h"
#include <string>

struct CSRMatrix {
  int n;
  int nnz;
  double* d_vals;
  int* d_row_ptr;
  int* d_col_idx;
  rocsparse_mat_descr descr;
  rocsparse_mat_info info;
};

struct PreconditionerData {
  std::string name;
  
  rocsparse_handle handle_rocsparse;
  
  // IC preconditioner data
  double* d_M_vals;
  rocsparse_mat_descr descrM;
  rocsparse_mat_descr descrL;
  rocsparse_mat_info infoM;
  void* buffer;
  size_t buffer_size;
  
  double* d_aux;
  
  int n;
  int nnz;
  int* d_row_ptr;
  int* d_col_idx;

  // Jacobi preconditioner data
  double* d_D;              // inverted diagonal: D[i] = 1.0 / A[i,i]
  const CSRMatrix* A_ptr;   // pointer to the matrix
  int jacobi_iter;          // number of Jacobi iterations (must be >= 1)
  double jacobi_omega;      // relaxation parameter (0 < omega <= 1)
  rocblas_handle handle_rocblas;

  // Asynch Jacobi specific
  int asynch_jacobi_version;  // version of asynch jacobi kernel (default 0)
};

int setup_preconditioner(const std::string& name,
                         const CSRMatrix& A,
                         PreconditionerData& precond_data,
                         rocsparse_handle handle_rocsparse);

int apply_preconditioner(const std::string& name,
                         double* d_x,
                         const double* d_r,
                         const CSRMatrix& A,
                         PreconditionerData& precond_data);

void cleanup_preconditioner(PreconditionerData& precond_data);

#endif
