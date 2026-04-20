#include "matrix_utils.h"
#include "preconditioner.h"
#include "pcg.h"
#define IT_MAX 100000

int main(int argc, char *argv[])
{
  int i;

  char* matrixFileName = NULL;
  char* rhsFileName = NULL;
  int maxit = 10;
  double tol = 1e-10;
  std::string preconditioner_name = "ic";

  int* h_A_coo_rows;
  int* h_A_coo_cols;
  double* h_A_coo_vals;

  int nn, nnnz, nm, nnz_or;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--matrix") == 0 && argc > i + 1) {
      matrixFileName = argv[++i];
    }
    if (strcmp(argv[i], "--rhs") == 0 && argc > i + 1) {
      rhsFileName = argv[++i];
    }
    if (strcmp(argv[i], "--tol") == 0 && argc > i + 1) {
      tol = atof(argv[++i]);
    } else if (strcmp(argv[i], "--maxit") == 0 && argc > i + 1) {
      maxit = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--precond") == 0 && argc > i + 1) {
      preconditioner_name = argv[++i];
    }
  }

  if (maxit > IT_MAX) {
    printf("\nWarning: setting maxit to IT_MAX = %d \n\n", IT_MAX);
    maxit = IT_MAX;
  }
  if (matrixFileName == NULL) {
    printf("Matrix file missing, exiting ... \n");
    return -1;
  }

  printf("Reading MTX file ... \n\n");

  read_mtx_file_into_coo(matrixFileName, &nn, &nm, &nnz_or, &nnnz, &h_A_coo_rows, &h_A_coo_cols, &h_A_coo_vals);

  double* h_A_csr_vals = (double*) calloc(nnnz, sizeof(double));
  int* h_A_csr_row_ptr = (int*) calloc(nn + 1, sizeof(int));
  int* h_A_csr_col_idx = (int*) calloc(nnnz, sizeof(int));

  coo_to_csr(nn,
             nnnz,
             h_A_coo_rows,
             h_A_coo_cols,
             h_A_coo_vals,
             h_A_csr_row_ptr,
             h_A_csr_col_idx,
             h_A_csr_vals);

  double* h_x = (double*) calloc(nn, sizeof(double));
  double* h_b = (double*) calloc(nn, sizeof(double));

  if (rhsFileName != NULL) {
    printf("Reading RHS file ... \n\n");
    read_rhs_file(rhsFileName, h_b);
  } else {
    printf("Creating RHS ...\n\n");
    for (i = 0; i < nn; ++i) {
      h_b[i] = 1.0;
    }
  }

  printf("Matrix info:\n");
  printf("\t Name:                                 %s\n", matrixFileName);
  printf("\t Size:                                 %d x %d\n", nn, nn);
  printf("\t Nnz (original):                       %d\n", nnz_or);
  printf("\t Nnz (expanded):                       %d\n", nnnz);
  if (rhsFileName != NULL) {
    printf("\t Rhs present?                          YES\n");
    printf("\t Rhs file:                             %s\n\n", rhsFileName);
  } else {
    printf("\t Rhs present?                          NO\n\n");
  }
  printf("CG setup: \n");
  printf("\t Maxit:                                %d\n", maxit);
  printf("\t Tolerance:                            %2.2e\n", tol);
  printf("\t Preconditioner:                       %s\n\n", preconditioner_name.c_str());

  CSRMatrix A;
  A.n = nn;
  A.nnz = nnnz;

  HIP_CHECK(hipMalloc(&A.d_row_ptr, sizeof(int) * (nn + 1)));
  HIP_CHECK(hipMalloc(&A.d_col_idx, sizeof(int) * nnnz));
  HIP_CHECK(hipMalloc(&A.d_vals, sizeof(double) * nnnz));

  HIP_CHECK(hipMemcpy(A.d_row_ptr, h_A_csr_row_ptr, sizeof(int) * (nn + 1), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(A.d_col_idx, h_A_csr_col_idx, sizeof(int) * nnnz, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(A.d_vals, h_A_csr_vals, sizeof(double) * nnnz, hipMemcpyHostToDevice));

  rocblas_handle handle_rocblas;
  rocsparse_handle handle_rocsparse;

  ROCBLAS_CHECK(rocblas_create_handle(&handle_rocblas));
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle_rocsparse));

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&A.descr));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(A.descr, rocsparse_index_base_zero));
  ROCSPARSE_CHECK(rocsparse_set_mat_type(A.descr, rocsparse_matrix_type_general));

  ROCSPARSE_CHECK(rocsparse_create_mat_info(&A.info));
  ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(handle_rocsparse,
                                            rocsparse_operation_none,
                                            nn,
                                            nn,
                                            nnnz,
                                            A.descr,
                                            A.d_vals,
                                            A.d_row_ptr,
                                            A.d_col_idx,
                                            A.info));

  printf("Initializing %s preconditioner ...\n\n", preconditioner_name.c_str());

  PreconditionerData precond_data;
  int setup_status = setup_preconditioner(preconditioner_name, A, precond_data, handle_rocsparse);
  if (setup_status != 0) {
    printf("Preconditioner setup failed. Exiting.\n");
    return -1;
  }

  printf("Preconditioner ready, starting CG ...\n\n");

  double* d_x;
  double* d_b;

  HIP_CHECK(hipMalloc(&d_x, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_b, sizeof(double) * nn));

  HIP_CHECK(hipMemset(d_x, 0, nn * sizeof(double)));
  HIP_CHECK(hipMemcpy(d_b, h_b, sizeof(double) * nn, hipMemcpyHostToDevice));

  PCGResult result = pcg_solve(A, d_x, d_b, maxit, tol,
                               preconditioner_name, precond_data,
                               handle_rocblas, handle_rocsparse);

  printf("\nCG summary results \n");
  printf("\t Iters              : %d  \n", result.iterations);
  printf("\t Res. norm          : %2.16g  \n", result.final_residual);
  if (result.flag == 0) {
    printf("\t Reason for exiting : CG converged  \n");
  } else if (result.flag == 1) {
    printf("\t Reason for exiting : CG reached maxit \n");
  } else {
    printf("\t Reason for exiting : CG failed\n");
  }
  printf("\n");

  free(h_A_coo_rows);
  free(h_A_coo_cols);
  free(h_A_coo_vals);
  free(h_A_csr_row_ptr);
  free(h_A_csr_col_idx);
  free(h_A_csr_vals);
  free(h_b);
  free(h_x);

  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(A.descr));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_info(A.info));

  cleanup_preconditioner(precond_data);

  HIP_CHECK(hipFree(A.d_vals));
  HIP_CHECK(hipFree(A.d_row_ptr));
  HIP_CHECK(hipFree(A.d_col_idx));

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_b));

  ROCBLAS_CHECK(rocblas_destroy_handle(handle_rocblas));
  ROCSPARSE_CHECK(rocsparse_destroy_handle(handle_rocsparse));

  return 0;
}
