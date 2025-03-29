#include "matrix_utils.h"
#define IT_MAX 100000

int main(int argc, char *argv[])
{
  int i ;

  char* matrixFileName = NULL;
  char* rhsFileName = NULL;
  int maxit = 10;
  double tol = 1e-10;

  int *h_A_coo_rows; 
  int *h_A_coo_cols;
  double *h_A_coo_vals;

  int nn, nnnz, nm, nnz_or; 

  for (int i = 1; i < argc; ++i) {
    if(strcmp(argv[i], "--matrix") == 0 && argc > i + 1)
    {
      matrixFileName = argv[++i];
    } 
    if(strcmp(argv[i], "--rhs") == 0 && argc > i + 1)
    {
      rhsFileName = argv[++i];
    } 
    if(strcmp(argv[i], "--tol") == 0 && argc > i + 1)
    {
      tol = atof(argv[++i]);
    } 
    else if(strcmp(argv[i], "--maxit") == 0 && argc > i + 1)
    {
      maxit = atoi(argv[++i]);
    } 
  } 

  if (maxit > IT_MAX) {
    printf("Warning: setting maxit to IT_MAX = %d \n\n", IT_MAX);
    maxit = IT_MAX;
  }
  if (matrixFileName == NULL) {
    printf("Matrix file missing, exiting ... \n");
    return -1;
  }

  printf("Reading MTX file ... \n\n");

  read_mtx_file_into_coo(matrixFileName, &nn, &nm, &nnz_or, &nnnz, &h_A_coo_rows, &h_A_coo_cols, &h_A_coo_vals);

  // allocate data for CSR

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
    for (i = 0; i < nn; ++i ) {
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
  printf("\t Tolerance:                            %2.2e\n\n", tol);


  // copy CSR matrix to the GPU    

  double* d_A_csr_vals;  
  int* d_A_csr_row_ptr;
  int* d_A_csr_col_idx;

  HIP_CHECK(hipMalloc(&d_A_csr_row_ptr, sizeof(int) * (nn + 1)));
  HIP_CHECK(hipMalloc(&d_A_csr_col_idx, sizeof(int) * nnnz));
  HIP_CHECK(hipMalloc(&d_A_csr_vals, sizeof(double) * nnnz));
  HIP_CHECK(hipMemcpy(d_A_csr_row_ptr, h_A_csr_row_ptr, sizeof(int) * (nn + 1), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_A_csr_col_idx, h_A_csr_col_idx, sizeof(int) * nnnz, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_A_csr_vals, h_A_csr_vals, sizeof(double) * nnnz, hipMemcpyHostToDevice));
  // before we call CG, we need to set up
  // a) matvec (buffer needs to be allocated)
  // b) incomplete Cholesky preconditioner

  // Library handles

  rocblas_handle handle_rocblas;
  rocsparse_handle  handle_rocsparse;

  ROCBLAS_CHECK(rocblas_create_handle(&handle_rocblas));
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle_rocsparse));

  // Matvec setup


  rocsparse_mat_descr descrA = NULL;
  rocsparse_mat_info  infoA;

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&(descrA)));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrA, rocsparse_index_base_zero));
  ROCSPARSE_CHECK(rocsparse_set_mat_type(descrA, rocsparse_matrix_type_general));

  ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoA));
  ROCSPARSE_CHECK(rocsparse_dcsrmv_analysis(handle_rocsparse,
					    rocsparse_operation_none,
					    nn,
					    nn,
					    nnnz,
					    descrA,
					    d_A_csr_vals,
					    d_A_csr_row_ptr,
					    d_A_csr_col_idx,
					    infoA));

  // Preconditioner (Incomple Cholesky)
  // descrM is needed for performing incomplete Cholesky.
  // descrLic, descrLtic are needed for triangular solves 
  printf("Initializing incomplete Cholesky preconditioner ...\n\n");
  /* since ic0 will over-write matrix values, we create a special array and copy them */
  double* d_M_csr_vals;
  HIP_CHECK(hipMalloc(&d_M_csr_vals, sizeof(double) * nnnz));
  HIP_CHECK(hipMemcpy(d_M_csr_vals, d_A_csr_vals, sizeof(double) * nnnz, hipMemcpyDeviceToDevice));

  rocsparse_mat_descr  descrM, descrLic, descrLtic; 
  void* ichol_buffer;

  rocsparse_mat_info infoM;

  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrM));
  ROCSPARSE_CHECK(rocsparse_set_mat_type(descrM, rocsparse_matrix_type_general));

  // for triangular solve with L
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrLic));
  ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descrLic, rocsparse_fill_mode_lower));
  ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descrLic, rocsparse_diag_type_non_unit));
  ROCSPARSE_CHECK(rocsparse_set_mat_index_base(descrLic, rocsparse_index_base_zero));

  /* for triangular solve with L^T */
  ROCSPARSE_CHECK(rocsparse_create_mat_descr(&descrLtic));
  ROCSPARSE_CHECK(rocsparse_set_mat_fill_mode(descrLtic, rocsparse_fill_mode_upper));
  ROCSPARSE_CHECK(rocsparse_set_mat_diag_type(descrLtic, rocsparse_diag_type_non_unit));
  ROCSPARSE_CHECK( rocsparse_set_mat_index_base(descrLtic, rocsparse_index_base_zero));

  ROCSPARSE_CHECK(rocsparse_create_mat_info(&infoM));

  /* Obtain required buffer size */
  size_t buffer_size_M;
  size_t buffer_size_L;
  size_t buffer_size_Lt;

  ROCSPARSE_CHECK(rocsparse_dcsric0_buffer_size(handle_rocsparse,
						nn,
						nnnz,
						descrM,
						d_M_csr_vals,
						d_A_csr_row_ptr,
						d_A_csr_col_idx,
						infoM,
						&buffer_size_M));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle_rocsparse,
					       rocsparse_operation_none,
					       nn,
					       nnnz,
					       descrLic,
					       d_M_csr_vals,
					       d_A_csr_row_ptr,
					       d_A_csr_col_idx,
					       infoM,
					       &buffer_size_L));

  ROCSPARSE_CHECK(rocsparse_dcsrsv_buffer_size(handle_rocsparse,
					       rocsparse_operation_transpose,
					       nn,
					       nnnz,
					       descrLic,
					       d_M_csr_vals,
					       d_A_csr_row_ptr,
					       d_A_csr_col_idx,
					       infoM,
					       &buffer_size_Lt));
  size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_Lt));
  // printf("finalsize %d \n",buffer_size);
  // Allocate temporary buffer
  HIP_CHECK(hipMalloc(&ichol_buffer, buffer_size));

  /* Perform analysis steps, using rocsparse_analysis_policy_reuse to improve 
   * computation performance */
  ROCSPARSE_CHECK(rocsparse_dcsric0_analysis(handle_rocsparse,
					     nn,
					     nnnz,
					     descrM,
					     d_M_csr_vals,
					     d_A_csr_row_ptr,
					     d_A_csr_col_idx,
					     infoM,
					     rocsparse_analysis_policy_reuse,
					     rocsparse_solve_policy_auto,
					     ichol_buffer));
  ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle_rocsparse,
					    rocsparse_operation_none,
					    nn,
					    nnnz,
					    descrLic,
					    d_M_csr_vals,
					    d_A_csr_row_ptr,
					    d_A_csr_col_idx,
					    infoM,
					    rocsparse_analysis_policy_reuse,
					    rocsparse_solve_policy_auto,
					    ichol_buffer));
  ROCSPARSE_CHECK(rocsparse_dcsrsv_analysis(handle_rocsparse,
					    rocsparse_operation_transpose,
					    nn,
					    nnnz,
					    descrLic,
					    d_M_csr_vals,
					    d_A_csr_row_ptr,
					    d_A_csr_col_idx,
					    infoM,
					    rocsparse_analysis_policy_reuse,
					    rocsparse_solve_policy_auto,
					    ichol_buffer));

  /* Check for zero pivot */
  rocsparse_int position;
  if (rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle_rocsparse,
								 infoM,
								 &position)) {
    printf("A has structural zero at A(%d,%d). Exiting.\n", position, position);
    return -1;
  }

  /* Compute incomplete Cholesky factorization M = LL' */
  ROCSPARSE_CHECK(rocsparse_dcsric0(handle_rocsparse,
				    nn,
				    nnnz,
				    descrM,
				    d_M_csr_vals,
				    d_A_csr_row_ptr,
				    d_A_csr_col_idx,
				    infoM,
				    rocsparse_solve_policy_auto,
				    ichol_buffer));
  // printf("status 3: %d \n", status_rocsparse);

  /* Check for zero pivot */
  if (rocsparse_status_zero_pivot == rocsparse_csric0_zero_pivot(handle_rocsparse,
								 infoM,
								 &position)) {
    printf("L has structural and/or numerical zero at L(%d,%d). Exiting.\n",
	   position,
	   position);
    return -1;
  }
  HIP_CHECK(hipDeviceSynchronize());

  printf("Preconditioner ready, starting CG ...\n\n");
  /* Actual CG */
  /* allocate and zero out  auxiliary arrays first */

  double* d_r;
  double* d_w;
  double* d_p;
  double* d_q;
  double* d_aux;

  HIP_CHECK(hipMalloc(&d_r, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_w, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_p, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_q, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_aux, sizeof(double) * nn));

  HIP_CHECK(hipMemset(d_r, 0, nn * sizeof(double)));
  HIP_CHECK(hipMemset(d_w, 0, nn * sizeof(double)));
  HIP_CHECK(hipMemset(d_p, 0, nn * sizeof(double)));
  HIP_CHECK(hipMemset(d_q, 0, nn * sizeof(double)));
  HIP_CHECK(hipMemset(d_aux, 0, nn * sizeof(double)));
  /* residual norm history */

  double* h_res_norm_history = (double*) calloc(maxit + 2 , sizeof(double));

  /* solution d_x and rhs d_b */
  double* d_x;
  double* d_b;

  HIP_CHECK(hipMalloc(&d_x, sizeof(double) * nn));
  HIP_CHECK(hipMalloc(&d_b, sizeof(double) * nn));
  /* x = 0 */
  HIP_CHECK(hipMemset(d_x, 0, nn * sizeof(double)));
  /* d_b = h_b */
  HIP_CHECK(hipMemcpy(d_b, h_b, sizeof(double) * nn, hipMemcpyHostToDevice));

  double alpha, beta, tolrel, rho_current, rho_previous, pTq;
  /* for matvecs and tri solves */
  const double one = 1.0;
  const double zero = 0.0;
  const double minusone = -1.0;
  int notconv = 1, iter = 0;

  //compute initial norm of r

  /* r = b - A*x */

  HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * nn, hipMemcpyDeviceToDevice));
  ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
				   rocsparse_operation_none,
				   nn,
				   nn,
				   nnnz,
				   &minusone,
				   descrA,
				   d_A_csr_vals,
				   d_A_csr_row_ptr,
				   d_A_csr_col_idx,
				   infoA,
				   d_x,
				   &one,
				   d_r));

  /* norm of r */

  ROCBLAS_CHECK(rocblas_ddot (handle_rocblas, 
			      nn, 
			      d_r, 
			      1, 
			      d_r, 
			      1, 
			      &h_res_norm_history[0]));

  h_res_norm_history[0] = sqrt(h_res_norm_history[0]);
  tolrel = tol * h_res_norm_history[0];

  printf("CG: it %d, res norm %5.5e \n", 0, h_res_norm_history[0]);
  int flag;
  int final_it;
  /* MAIN LOOP */
  while (notconv) {
    HIP_CHECK(hipMemset(d_w, 0, nn * sizeof(double)));
    
    /* w = ichol(r) */

    /* phase 1: d_aux = L\r */
    ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle_rocsparse,
					   rocsparse_operation_none,
					   nn,
					   nnnz,
					   &one,
					   descrLic,
					   d_M_csr_vals,
					   d_A_csr_row_ptr,
					   d_A_csr_col_idx,
					   infoM,
					   d_r,//input
					   d_aux, //output 
					   rocsparse_solve_policy_auto,
					   ichol_buffer));

    /* phase 2: d_w = L^T\aux */

    ROCSPARSE_CHECK(rocsparse_dcsrsv_solve(handle_rocsparse,
					   rocsparse_operation_transpose,
					   nn,
					   nnnz,
					   &one,
					   descrLic,
					   d_M_csr_vals,
					   d_A_csr_row_ptr,
					   d_A_csr_col_idx,
					   infoM,
					   d_aux, // input 
					   d_w,  // output
					   rocsparse_solve_policy_auto,
					   ichol_buffer));
    HIP_CHECK(hipDeviceSynchronize());
      HIP_CHECK(hipMemcpy(d_w, d_r, sizeof(double) * nn, hipMemcpyDeviceToDevice));
    /* rho_current = r'*w; */
    ROCBLAS_CHECK(rocblas_ddot (handle_rocblas, 
				nn, 
				d_r, 
				1, 
				d_w, 
				1, 
				&rho_current));
    //    rho_current = dot(n, r, w);
    if (iter == 0) {
      /* p = w */
      //    vec_copy(n, w, p);
      HIP_CHECK(hipMemcpy(d_p, d_w, sizeof(double) * nn, hipMemcpyDeviceToDevice));
    } else {
      beta = rho_current/rho_previous;
      /* p = w+bet*p; */

      ROCBLAS_CHECK(rocblas_dscal(handle_rocblas, 
				  nn,
				  &beta,
				  d_p, 
				  1));

      ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas, 
				  nn,
				  &one,
				  d_w, 
				  1,
				  d_p, 
				  1));
      HIP_CHECK(hipDeviceSynchronize());
    }
    /* q = A*p; */

    ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
				     rocsparse_operation_none,
				     nn,
				     nn,
				     nnnz,
				     &one,
				     descrA,
				     d_A_csr_vals,
				     d_A_csr_row_ptr,
				     d_A_csr_col_idx,
				     infoA,
				     d_p,
				     &zero,
				     d_q));

    /* alpha = rho_current/(p'*q);*/

    ROCBLAS_CHECK(rocblas_ddot (handle_rocblas, 
				nn, 
				d_p, 
				1, 
				d_q, 
				1, 
				&pTq));
    alpha = rho_current / pTq; 

    /* x = x + alph*p; */

    ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas, 
				nn,
				&alpha,
				d_p, 
				1,
				d_x, 
				1));
    /* r = r - alph*q; */
    alpha *= (-1.0);
   ROCBLAS_CHECK(rocblas_daxpy(handle_rocblas, 
				nn,
				&alpha,
				d_q, 
				1,
				d_r, 
				1));

    alpha *= (-1.0);
    /* norm of r */
    iter++;
    ROCBLAS_CHECK(rocblas_ddot (handle_rocblas, 
				nn, 
				d_r, 
				1, 
				d_r, 
				1, 
				&h_res_norm_history[iter]));

    h_res_norm_history[iter] = sqrt(h_res_norm_history[iter]);
    printf("CG: it %d, res norm %5.16e \n",iter, h_res_norm_history[iter]);

    /* check convergence */
    if ((h_res_norm_history[iter]) < tolrel) {
      flag = 0;
      notconv = 0;
      final_it = iter; 

      /* r = b - A*x */

      HIP_CHECK(hipMemcpy(d_r, d_b, sizeof(double) * nn, hipMemcpyDeviceToDevice));
      ROCSPARSE_CHECK(rocsparse_dcsrmv(handle_rocsparse,
				       rocsparse_operation_none,
				       nn,
				       nn,
				       nnnz,
				       &minusone,
				       descrA,
				       d_A_csr_vals,
				       d_A_csr_row_ptr,
				       d_A_csr_col_idx,
				       infoA,
				       d_x,
				       &one,
				       d_r));
      double r2_final;
      ROCBLAS_CHECK(rocblas_ddot (handle_rocblas, 
				  nn, 
				  d_r, 
				  1, 
				  d_r, 
				  1, 
				  &r2_final));
      printf("TRUE Norm of r %5.16e\n", sqrt(r2_final));  
    } else {
      if (iter > maxit){
	flag = 1;
	notconv = 0;
	final_it = iter; 
      }
    }
    rho_previous = rho_current;
  }//while

  printf("CG summary results \n");
  printf("\t Iters              : %d  \n", final_it);
  printf("\t Res. norm          : %2.16g  \n", h_res_norm_history[final_it]);
  if (flag == 0){
    printf("\t Reason for exiting : CG converged  \n");
  } else {
    if (flag == 1){
      printf("\t Reason for exiting : CG reached maxit \n");
    } else {
      printf("\t Reason for exiting : CG failed\n");
    }
  }
  // cleanup

  // CPU
  free(h_A_coo_rows); 
  free(h_A_coo_cols); 
  free(h_A_coo_vals);
  free(h_A_csr_row_ptr); 
  free(h_A_csr_col_idx); 
  free(h_A_csr_vals);
  free(h_b);
  free(h_x);
  free(h_res_norm_history);  
  // GPU
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrA));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_info(infoA));

  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrM));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrLic));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_descr(descrLtic));
  ROCSPARSE_CHECK(rocsparse_destroy_mat_info(infoM));
  HIP_CHECK(hipFree(ichol_buffer));  

  HIP_CHECK(hipFree(d_A_csr_vals));  
  HIP_CHECK(hipFree(d_A_csr_row_ptr));
  HIP_CHECK(hipFree(d_A_csr_col_idx));

  HIP_CHECK(hipFree(d_M_csr_vals));  

  HIP_CHECK(hipFree(d_r));
  HIP_CHECK(hipFree(d_w));
  HIP_CHECK(hipFree(d_p));
  HIP_CHECK(hipFree(d_q));

  HIP_CHECK(hipFree(d_x));
  HIP_CHECK(hipFree(d_b));
 return 0;
}
