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
  
  // Modern rocsparse_v2_spmv API
  rocsparse_spmat_descr spmat;
  rocsparse_spmv_descr spmv_descr;
  void* spmv_buffer;
  size_t spmv_buffer_size;
  
  // Legacy (for IC preconditioner which still uses csrsv)
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

  // Gauss-Seidel preconditioner data
  // Lower triangular L (strictly lower)
  double* d_L_vals;
  int* d_L_row_ptr;
  int* d_L_col_idx;
  int L_nnz;
  rocsparse_spmat_descr spmatL;
  rocsparse_spmv_descr spmv_descr_L;
  void* spmv_buffer_L;
  size_t spmv_buffer_size_L;

  // Upper triangular U (strictly upper)
  double* d_U_vals;
  int* d_U_row_ptr;
  int* d_U_col_idx;
  int U_nnz;
  rocsparse_spmat_descr spmatU;
  rocsparse_spmv_descr spmv_descr_U;
  void* spmv_buffer_U;
  size_t spmv_buffer_size_U;

  // Auxiliary vectors for GS
  double* d_aux_vec1;
  double* d_aux_vec2;
  double* d_aux_vec3;

  // GS iteration parameters
  int gs_inner_iter;  // k - inner iterations
  int gs_outer_iter;  // m - outer iterations

  // Device scalars for fully async operations
  double* d_one;
  double* d_zero;
  double* d_minusone;
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
