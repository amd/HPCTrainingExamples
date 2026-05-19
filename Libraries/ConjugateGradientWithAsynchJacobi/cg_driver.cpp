ginclude "matrix_utils.h"
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
  int jacobi_iter = 3;
  double jacobi_omega = 0.67;
  int asynch_jacobi_version = 0;
  int gs_inner_iter = 3;
  int gs_outer_iter = 1;

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
    } else if (strcmp(argv[i], "--jacobi_iter") == 0 && argc > i + 1) {
      jacobi_iter = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--jacobi_omega") == 0 && argc > i + 1) {
      jacobi_omega = atof(argv[++i]);
    } else if (strcmp(argv[i], "--asynch_jacobi_version") == 0 && argc > i + 1) {
      asynch_jacobi_version = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--gs_inner_iter") == 0 && argc > i + 1) {
      gs_inner_iter = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--gs_outer_iter") == 0 && argc > i + 1) {
      gs_outer_iter = atoi(argv[++i]);
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
    printf("Creating RHS as b = A * ones ...\n\n");
    for (i = 0; i < nn; ++i) {
      h_b[i] = 0.0;
      for (int j = h_A_csr_row_ptr[i]; j < h_A_csr_row_ptr[i + 1]; ++j) {
        h_b[i] += h_A_csr_vals[j];
      }
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
  printf("\t Preconditioner:                       %s\n", preconditioner_name.c_str());
  if (preconditioner_name == "jacobi" || preconditioner_name == "asynch_jacobi") {
    printf("\t Jacobi iterations:                    %d\n", jacobi_iter);
    printf("\t Jacobi omega:                         %g\n", jacobi_omega);
  }
  if (preconditioner_name == "asynch_jacobi") {
    printf("\t Asynch Jacobi version:                %d\n", asynch_jacobi_version);
  }
  if (preconditioner_name == "gs_it" || preconditioner_name == "gs_it2") {
    printf("\t GS inner iterations (k):              %d\n", gs_inner_iter);
    printf("\t GS outer iterations (m):              %d\n", gs_outer_iter);
  }
  printf("\n");

  CSRMatrix A = {};  // Zero-initialize all members
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

  // Set device pointer mode for both handles
  ROCBLAS_CHECK(rocblas_set_pointer_mode(handle_rocblas, rocblas_pointer_mode_device));
  ROCSPARSE_CHECK(rocsparse_set_pointer_mode(handle_rocsparse, rocsparse_pointer_mode_device));

  // Legacy descriptors (for IC preconditioner)
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&A.descr));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(A.descr, rocsparse_index_base_zero));
  ROCSPARSE_CHECK(rocsparse_set_mat_type(A.descr, rocsparse_matrix_type_general));
  ROCSPARSE_CHECK(rocsparse_create_mat_info(&A.info));

  // Modern rocsparse_v2_spmv setup
  ROCSPARSE_CHECK(rocsparse_create_csr_descr(&A.spmat,
                                             nn,
                                             nn,
                                             nnnz,
                                             A.d_row_ptr,
                                             A.d_col_idx,
                                             A.d_vals,
                                             rocsparse_indextype_i32,
                                             rocsparse_indextype_i32,
                                             rocsparse_index_base_zero,
                                             rocsparse_datatype_f64_r));

  // Create spmv descriptor and set all required inputs
  ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&A.spmv_descr));
  rocsparse_operation op = rocsparse_operation_none;
  rocsparse_spmv_alg alg = rocsparse_spmv_alg_default;
  rocsparse_datatype scalar_type = rocsparse_datatype_f64_r;
  rocsparse_datatype compute_type = rocsparse_datatype_f64_r;
  rocsparse_error set_input_error;
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           A.spmv_descr,
                                           rocsparse_spmv_input_operation,
                                           &op,
                                           sizeof(op),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           A.spmv_descr,
                                           rocsparse_spmv_input_alg,
                                           &alg,
                                           sizeof(alg),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           A.spmv_descr,
                                           rocsparse_spmv_input_scalar_datatype,
                                           &scalar_type,
                                           sizeof(scalar_type),
                                           &set_input_error));
  ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle_rocsparse,
                                           A.spmv_descr,
                                           rocsparse_spmv_input_compute_datatype,
                                           &compute_type,
                                           sizeof(compute_type),
                                           &set_input_error));

  // Create temporary vectors for buffer size query
  rocsparse_dnvec_descr tmp_x, tmp_y;
  double* d_tmp;
  HIP_CHECK(hipMalloc(&d_tmp, sizeof(double) * nn));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&tmp_x,
                                               nn,
                                               d_tmp,
                                               rocsparse_datatype_f64_r));
  ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(&tmp_y,
                                               nn,
                                               d_tmp,
                                               rocsparse_datatype_f64_r));

  // Get buffer size for analysis stage
  rocsparse_error spmv_error;
  ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(handle_rocsparse,
                                                A.spmv_descr,
                                                A.spmat,
                                                tmp_x,
                                                tmp_y,
                                                rocsparse_v2_spmv_stage_analysis,
                                                &A.spmv_buffer_size,
                                                &spmv_error));

  if (A.spmv_buffer_size == 0) {
    A.spmv_buffer = nullptr;
  } else {
    HIP_CHECK(hipMalloc(&A.spmv_buffer, A.spmv_buffer_size));
  }

  // Perform analysis
  double h_alpha = 1.0, h_beta = 0.0;
  ROCSPARSE_CHECK(rocsparse_v2_spmv(handle_rocsparse,
                                    A.spmv_descr,
                                    &h_alpha,
                                    A.spmat,
                                    tmp_x,
                                    &h_beta,
                                    tmp_y,
                                    rocsparse_v2_spmv_stage_analysis,
                                    A.spmv_buffer_size,
                                    A.spmv_buffer,
                                    &spmv_error));

  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(tmp_x));
  ROCSPARSE_CHECK(rocsparse_destroy_dnvec_descr(tmp_y));
  HIP_CHECK(hipFree(d_tmp));

  printf("Initializing %s preconditioner ...\n\n", preconditioner_name.c_str());

  PreconditionerData precond_data = {};  // Zero-initialize all POD members
  precond_data.jacobi_iter = jacobi_iter;
  precond_data.jacobi_omega = jacobi_omega;
  precond_data.asynch_jacobi_version = asynch_jacobi_version;
  precond_data.gs_inner_iter = gs_inner_iter;
  precond_data.gs_outer_iter = gs_outer_iter;
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

  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));

  HIP_CHECK(hipEventRecord(start));

  PCGResult result = pcg_solve(A, d_x, d_b, maxit, tol,
                               preconditioner_name, precond_data,
                               handle_rocblas, handle_rocsparse);

  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&elapsed_ms, start, stop));

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));

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
  printf("\t Time (ms)          : %.3f\n", elapsed_ms);
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
  ROCSPARSE_CHECK(rocsparse_destroy_spmat_descr(A.spmat));
  ROCSPARSE_CHECK(rocsparse_destroy_spmv_descr(A.spmv_descr));
  HIP_CHECK(hipFree(A.spmv_buffer));

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
