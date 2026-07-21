#include "klu.h"
#include "umfpack.h" // we will use umfpack for conversion from coo
#include "matrix_utils.h"

#include "thrust/host_vector.h"
#include "thrust/device_vector.h"
#include "thrust/copy.h" 
#include <thrust/extrema.h>

int main(int argc, char *argv[])
{
  double *null = (double *) NULL ;
  int i ;

  char* matrixFileName1 = NULL;
  char* matrixFileName2 = NULL;
  char* matrixFileName3 = NULL;
  char* rhsFileName1 = NULL;
  char* rhsFileName2 = NULL;
  char* rhsFileName3 = NULL;
  int* h_A_coo_rows; 
  int *h_A_coo_cols;
  double *h_A_coo_vals;
  int nn, nnnz, nm; 
  // Read options: use --matrix1, --matrix2, --matrix3, --rhs1, --rhs2, --rhs3 

  for (int i = 1; i < argc; ++i) {
    if(strcmp(argv[i], "--matrix1") == 0 && argc > i + 1)
    {
      matrixFileName1 = argv[++i];
    } 
    if(strcmp(argv[i], "--matrix2") == 0 && argc > i + 1)
    {
      matrixFileName2 = argv[++i];
    } 
    if(strcmp(argv[i], "--matrix3") == 0 && argc > i + 1)
    {
      matrixFileName3 = argv[++i];
    } 
    if(strcmp(argv[i], "--rhs1") == 0 && argc > i + 1)
    {
      rhsFileName1 = argv[++i];
    } 
    if(strcmp(argv[i], "--rhs2") == 0 && argc > i + 1)
    {
      rhsFileName2 = argv[++i];
    } 
    else if(strcmp(argv[i], "--rhs3") == 0 && argc > i + 1)
    {
      rhsFileName3 = argv[++i];
    } 
  } 


  if (matrixFileName1 == NULL) {
    printf("First matrix missing, exiting ... \n");
    return -1;
  }

  printf("\n\nReading first MTX file ... \n");

  read_mtx_file_into_coo(matrixFileName1, &nn, &nm, &nnnz, &h_A_coo_rows, &h_A_coo_cols, &h_A_coo_vals);

  thrust::host_vector<double> h_x(nn);
  thrust::host_vector<double> h_b(nn);

  if (rhsFileName1 != NULL) {
    printf("Reading RHS file ... \n");
    read_rhs_file(rhsFileName1, thrust::raw_pointer_cast(&h_b[0]));
  } else {
    printf("Creating RHS ...\n");
    thrust::fill(thrust::host, h_b.begin(), h_b.end(), 1.0);
  }

  printf("\n\n============================== \n\n");
  printf("First matrix info:\n");
  printf("\t Name:                                 %s\n", matrixFileName1);
  printf("\t Size:                                 %d x %d\n", nn, nn);
  printf("\t Nnz (expanded):                       %d\n", nnnz);
  if (rhsFileName1 != NULL) {
    printf("\t Rhs present?                          YES\n");
    printf("\t Rhs file:                             %s\n", rhsFileName1);
  } else {
    printf("\t Rhs present?                          NO\n");
  }
  printf("\n\n============================== \n\n");


  thrust::host_vector<double> h_A_csc_vals(nnnz);
  thrust::host_vector<int> h_A_csc_col_pointers(nn + 1);
  thrust::host_vector<int> h_A_csc_row_idx(nnnz);
  // note that 
  // now coo2csc
  int status;
  // note: this can be done using rocsparse, however, since data comes on the CPU, and KLU requires CPU data, it makes more sense to do it this way.
  status = umfpack_di_triplet_to_col(nn,
                                     nm,
                                     nnnz,
                                     h_A_coo_rows,
                                     h_A_coo_cols,
                                     h_A_coo_vals,
                                     thrust::raw_pointer_cast(&h_A_csc_col_pointers[0]),
                                     thrust::raw_pointer_cast(&h_A_csc_row_idx[0]),
                                     thrust::raw_pointer_cast(&h_A_csc_vals[0]), 
                                     NULL);
  // status == 0 when thigs are OK
  if (status != 0)  {
    printf("COO 2 CSC returned status %d \n", status);
  }
  // use KLU from here
  klu_symbolic* Symbolic;
  klu_numeric* Numeric;
  klu_common Common;
  // set default params
  klu_defaults (&Common);

  // If refactorization doesnt work too well, experiment with this values
  Common.ordering = 1; // 0 = AMD, 1 = COLAMD 
                       //  Common.tol = 1.0; 
                       //  Common.btf = 1;
  Common.halt_if_singular = 0; // for matrices with very high condition number

  // note: values not given
  Symbolic = klu_analyze(nn, 
                         thrust::raw_pointer_cast(&h_A_csc_col_pointers[0]),
                         thrust::raw_pointer_cast(&h_A_csc_row_idx[0]),
                         &Common);

  if (Symbolic == NULL) {
    printf("KLU Symbolic failed");
  }

  Numeric = klu_factor (thrust::raw_pointer_cast(&h_A_csc_col_pointers[0]),
                        thrust::raw_pointer_cast(&h_A_csc_row_idx[0]),
                        thrust::raw_pointer_cast(&h_A_csc_vals[0]), 
                        Symbolic, 
                        &Common);

  if (Numeric == NULL) {
    printf("KLU Numeric failed");
  }

  // copy b to x as solve will OVERWRITE the vector 

  thrust::copy(thrust::host, h_b.begin(), h_b.end(), h_x.begin() );

  status = klu_solve (Symbolic, Numeric,nn, 1,  thrust::raw_pointer_cast(&h_x[0]), &Common);

  //error codes are reversed between KLU and UMFPACK ...
  if (status != 1)  {
    printf("Solve failed with status %d \n", status);
  }

  int nnzL = Numeric->lnz;
  int nnzU = Numeric->unz;

  thrust::host_vector<double> h_L_csc_vals(nnzL);
  thrust::host_vector<int> h_L_csc_col_pointers(nn + 1);
  thrust::host_vector<int> h_L_csc_row_idx(nnzL);

  thrust::host_vector<double> h_U_csc_vals(nnzU);
  thrust::host_vector<int> h_U_csc_col_pointers(nn + 1);
  thrust::host_vector<int> h_U_csc_row_idx(nnzU);


  thrust::host_vector<int> h_P(nn);
  thrust::host_vector<int> h_Q(nn);

  memcpy(thrust::raw_pointer_cast(&h_P[0]), Numeric->Pnum, sizeof(int) * nn);
  memcpy(thrust::raw_pointer_cast(&h_Q[0]), Symbolic->Q, sizeof(int) * nn);

  // Now get L, U from KLU. KLU stores both L and U as CSC ... 
  // note: this is not a way to get P and Q (see above)
  status = klu_extract(Numeric, 
                       Symbolic, 
                       thrust::raw_pointer_cast(&h_L_csc_col_pointers[0]),
                       thrust::raw_pointer_cast(&h_L_csc_row_idx[0]),
                       thrust::raw_pointer_cast(&h_L_csc_vals[0]),
                       thrust::raw_pointer_cast(&h_U_csc_col_pointers[0]),
                       thrust::raw_pointer_cast(&h_U_csc_row_idx[0]),
                       thrust::raw_pointer_cast(&h_U_csc_vals[0]),
                       NULL, 
                       NULL, 
                       NULL,
                       NULL, 
                       NULL,
                       NULL, 
                       NULL, 
                       &Common);

  if (status != 1)  {
    printf("klu_extract returned status %d \n", status);
  } else {

    printf("KLU LU factorization info:\n");
    printf("\t Size:                                 %d x %d \n", nn, nn);
    printf("\t L nnz:                                %d\n", nnzL);
    printf("\t U nnz:                                %d\n", nnzU);
    printf("\n\n============================== \n\n");
  }

  // Now transpose L and U to put them in CSR format (use rocsparse)

  thrust::device_vector<double> d_L_csc_vals{h_L_csc_vals};
  thrust::device_vector<int> d_L_csc_col_pointers{h_L_csc_col_pointers};
  thrust::device_vector<int> d_L_csc_row_idx{h_L_csc_row_idx};

  thrust::device_vector<double> d_L_csr_vals(nnzL);
  thrust::device_vector<int> d_L_csr_row_pointers(nn + 1);
  thrust::device_vector<int> d_L_csr_col_idx(nnzL);

  thrust::device_vector<double> d_U_csc_vals{h_U_csc_vals};
  thrust::device_vector<int> d_U_csc_col_pointers{h_U_csc_col_pointers};
  thrust::device_vector<int> d_U_csc_row_idx{h_U_csc_row_idx};

  thrust::device_vector<double> d_U_csr_vals(nnzU);
  thrust::device_vector<int> d_U_csr_row_pointers(nn + 1);
  thrust::device_vector<int> d_U_csr_col_idx(nnzU);

  // The same but for A

  thrust::device_vector<double> d_A_csc_vals{h_A_csc_vals};
  thrust::device_vector<int> d_A_csc_col_pointers{h_A_csc_col_pointers};
  thrust::device_vector<int> d_A_csc_row_idx{h_A_csc_row_idx};

  thrust::device_vector<double> d_A_csr_vals(nnnz);
  thrust::device_vector<int> d_A_csr_row_pointers(nn + 1);
  thrust::device_vector<int> d_A_csr_col_idx(nnnz);


  // check for buffer sizes for CSC2CSR, we can use one buffer but lets make sure it is sufficiently large
  rocblas_handle handle_rocblas;
  rocsparse_handle handle_rocsparse; 
  ROCBLAS_CHECK(rocblas_create_handle(&handle_rocblas));  
  ROCSPARSE_CHECK(rocsparse_create_handle(&handle_rocsparse));

  size_t buffer_size;
  size_t buffer_size_U, buffer_size_L,  buffer_size_A;
  void* d_csc2csr_buffer;

  ROCSPARSE_CHECK(rocsparse_csr2csc_buffer_size(handle_rocsparse,
                                                nn,
                                                nn,
                                                nnzL,
                                                thrust::raw_pointer_cast(&d_L_csc_col_pointers[0]),
                                                thrust::raw_pointer_cast(&d_L_csc_row_idx[0]),
                                                rocsparse_action_numeric,
                                                &buffer_size_L));

  HIP_CHECK(hipDeviceSynchronize());
  ROCSPARSE_CHECK(rocsparse_csr2csc_buffer_size(handle_rocsparse,
                                                nn,
                                                nn,
                                                nnzU,
                                                thrust::raw_pointer_cast(&d_U_csc_col_pointers[0]),
                                                thrust::raw_pointer_cast(&d_U_csc_row_idx[0]),
                                                rocsparse_action_numeric,
                                                &buffer_size_U));

  HIP_CHECK(hipDeviceSynchronize());
  ROCSPARSE_CHECK(rocsparse_csr2csc_buffer_size(handle_rocsparse,
                                                nn,
                                                nn,
                                                nnnz,
                                                thrust::raw_pointer_cast(&d_A_csc_col_pointers[0]),
                                                thrust::raw_pointer_cast(&d_A_csc_row_idx[0]),
                                                rocsparse_action_numeric,
                                                &buffer_size_A));
  buffer_size = std::max(buffer_size_A, std::max(buffer_size_L, buffer_size_U));

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMalloc(&d_csc2csr_buffer, buffer_size));

  // Now transpose L and U to put them in CSR format
  ROCSPARSE_CHECK(rocsparse_dcsr2csc(handle_rocsparse,
                                     nn,
                                     nn,
                                     nnzL,
                                     thrust::raw_pointer_cast(&d_L_csc_vals[0]),
                                     thrust::raw_pointer_cast(&d_L_csc_col_pointers[0]),
                                     thrust::raw_pointer_cast(&d_L_csc_row_idx[0]),
                                     thrust::raw_pointer_cast(&d_L_csr_vals[0]),
                                     thrust::raw_pointer_cast(&d_L_csr_col_idx[0]),
                                     thrust::raw_pointer_cast(&d_L_csr_row_pointers[0]),
                                     rocsparse_action_numeric,
                                     rocsparse_index_base_zero,
                                     d_csc2csr_buffer));
  HIP_CHECK(hipDeviceSynchronize());

  ROCSPARSE_CHECK(rocsparse_dcsr2csc(handle_rocsparse,
                                     nn,
                                     nn,
                                     nnzU,
                                     thrust::raw_pointer_cast(&d_U_csc_vals[0]),
                                     thrust::raw_pointer_cast(&d_U_csc_col_pointers[0]),
                                     thrust::raw_pointer_cast(&d_U_csc_row_idx[0]),
                                     thrust::raw_pointer_cast(&d_U_csr_vals[0]),
                                     thrust::raw_pointer_cast(&d_U_csr_col_idx[0]),
                                     thrust::raw_pointer_cast(&d_U_csr_row_pointers[0]),
                                     rocsparse_action_numeric,
                                     rocsparse_index_base_zero,
                                     d_csc2csr_buffer));
  HIP_CHECK(hipDeviceSynchronize());
  // transpose A too

  ROCSPARSE_CHECK(rocsparse_dcsr2csc(handle_rocsparse,
                                     nn,
                                     nn,
                                     nnnz,
                                     thrust::raw_pointer_cast(&d_A_csc_vals[0]),
                                     thrust::raw_pointer_cast(&d_A_csc_col_pointers[0]),
                                     thrust::raw_pointer_cast(&d_A_csc_row_idx[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_col_idx[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]),
                                     rocsparse_action_numeric,
                                     rocsparse_index_base_zero,
                                     d_csc2csr_buffer));

  HIP_CHECK(hipDeviceSynchronize());

  // Now both L and U are in CSR and live on the GPU-- time to set up Refactorization

  thrust::device_vector<int> d_P{h_P};
  thrust::device_vector<int> d_Q{h_Q};

  // Also d_x, d_b

  thrust::device_vector<double> d_x{h_x};
  thrust::device_vector<double> d_b{h_b};

  // check how good (or bad) x is

  thrust::device_vector<double> d_res(nn);

  // use formula from Arioli, Demmel, Duff (1989)

  backward_error_estimate(nn,
                          thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]), 
                          thrust::raw_pointer_cast(&d_A_csr_col_idx[0]), 
                          thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                          thrust::raw_pointer_cast(&d_x[0]),
                          thrust::raw_pointer_cast(&d_b[0]),
                          thrust::raw_pointer_cast(&d_res[0]));
  HIP_CHECK(hipDeviceSynchronize());

  // now we need vector d_res inf norm 
  thrust::device_vector<double>::iterator nrm_it = thrust::max_element(d_res.begin(), d_res.end(), abs_compare());

  double omega = *nrm_it; 
  printf("First system: KLU backward error (computed on the GPU): %16.16e\n", omega);

  // note: rocsolver uses rocblas handle 

  rocsolver_rfinfo infoM;
  ROCBLAS_CHECK(rocsolver_create_rfinfo(&infoM, handle_rocblas));

  // create M = L + U - I
  int nnzM = nnzL + nnzU - nn;
  thrust::device_vector<double> d_M_csr_vals(nnzM);
  thrust::device_vector<int> d_M_csr_row_pointers(nn + 1);
  thrust::device_vector<int> d_M_csr_col_idx(nnzM);


  // we need to add the factors together.  
  ROCBLAS_CHECK(rocsolver_dcsrrf_sumlu(handle_rocblas, 
                                       nn, 
                                       nnzL, 
                                       thrust::raw_pointer_cast(&d_L_csr_row_pointers[0]), 
                                       thrust::raw_pointer_cast(&d_L_csr_col_idx[0]), 
                                       thrust::raw_pointer_cast(&d_L_csr_vals[0]),
                                       nnzU, 
                                       thrust::raw_pointer_cast(&d_U_csr_row_pointers[0]), 
                                       thrust::raw_pointer_cast(&d_U_csr_col_idx[0]), 
                                       thrust::raw_pointer_cast(&d_U_csr_vals[0]),
                                       thrust::raw_pointer_cast(&d_M_csr_row_pointers[0]), 
                                       thrust::raw_pointer_cast(&d_M_csr_col_idx[0]), 
                                       thrust::raw_pointer_cast(&d_M_csr_vals[0])));

  HIP_CHECK(hipDeviceSynchronize());
  // analyze what we've got
  ROCBLAS_CHECK(rocsolver_dcsrrf_analysis(handle_rocblas,
                                          nn,
                                          1,
                                          nnnz,
                                          thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]), 
                                          thrust::raw_pointer_cast(&d_A_csr_col_idx[0]), 
                                          thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                                          nnzM,
                                          thrust::raw_pointer_cast(&d_M_csr_row_pointers[0]), 
                                          thrust::raw_pointer_cast(&d_M_csr_col_idx[0]), 
                                          thrust::raw_pointer_cast(&d_M_csr_vals[0]),
                                          thrust::raw_pointer_cast(&d_P[0]),
                                          thrust::raw_pointer_cast(&d_Q[0]),
                                          thrust::raw_pointer_cast(&d_b[0]),
                                          nn,
                                          infoM));
  HIP_CHECK(hipDeviceSynchronize());

  // Read new A and new rhs
  if (matrixFileName2 == NULL) {
    printf("\n");
    printf("Second matrix NOT provided, exiting ... \n");
    return -1;
  }
  // 
  // You do not have to do it in an actual application, but the matrix reading routine ALLOCs these again so...
  free(h_A_coo_rows);
  free(h_A_coo_cols);
  free(h_A_coo_vals);

  printf("\n\nReading second MTX file ... \n");

  read_mtx_file_into_coo(matrixFileName2, &nn, &nm, &nnnz, &h_A_coo_rows, &h_A_coo_cols, &h_A_coo_vals);

  if (rhsFileName2 != NULL) {
    printf("Reading RHS file ... \n");
    read_rhs_file(rhsFileName2, thrust::raw_pointer_cast(&h_b[0]));
  } else {
    printf("Creating RHS ...\n");
    thrust::fill(thrust::host, h_b.begin(), h_b.end(), 1.0);
  }

  thrust::copy(thrust::device, h_b.begin(), h_b.end(), d_b.begin() );
  printf("\n\n============================== \n\n");
  printf("Second matrix info:\n");
  printf("\t Name:                                 %s\n", matrixFileName2);
  printf("\t Size:                                 %d x %d\n", nn, nn);
  printf("\t Nnz (expanded):                       %d\n", nnnz);
  if (rhsFileName2 != NULL) {
    printf("\t Rhs present?                          YES\n");
    printf("\t Rhs file:                             %s\n", rhsFileName2);
  } else {
    printf("\t Rhs present?                          NO\n");
  }
  printf("\n\n============================== \n\n");

  // COO 2 CSC
  status = umfpack_di_triplet_to_col(nn,
                                     nm,
                                     nnnz,
                                     h_A_coo_rows,
                                     h_A_coo_cols,
                                     h_A_coo_vals,
                                     thrust::raw_pointer_cast(&h_A_csc_col_pointers[0]),
                                     thrust::raw_pointer_cast(&h_A_csc_row_idx[0]),
                                     thrust::raw_pointer_cast(&h_A_csc_vals[0]), 
                                     NULL);
  // status == 0 when thigs are OK
  if (status != 0)  {
    printf("COO 2 CSC returned status %d \n", status);
  }

  // update on the GPU
  thrust::copy(thrust::device, h_A_csc_col_pointers.begin(), h_A_csc_col_pointers.end(), d_A_csc_col_pointers.begin() );
  thrust::copy(thrust::device, h_A_csc_row_idx.begin(), h_A_csc_row_idx.end(), d_A_csc_row_idx.begin() );
  thrust::copy(thrust::device, h_A_csc_vals.begin(), h_A_csc_vals.end(), d_A_csc_vals.begin() );


  // NOW CSC 2 CSR

  HIP_CHECK(hipDeviceSynchronize());
  ROCSPARSE_CHECK(rocsparse_dcsr2csc(handle_rocsparse,
                                     nn,
                                     nn,
                                     nnnz,
                                     thrust::raw_pointer_cast(&d_A_csc_vals[0]),
                                     thrust::raw_pointer_cast(&d_A_csc_col_pointers[0]),
                                     thrust::raw_pointer_cast(&d_A_csc_row_idx[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_col_idx[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]),
                                     rocsparse_action_numeric,
                                     rocsparse_index_base_zero,
                                     d_csc2csr_buffer));

  HIP_CHECK(hipDeviceSynchronize());

  ROCBLAS_CHECK(rocsolver_dcsrrf_refactlu(handle_rocblas,
                                          nn,
                                          nnnz,
                                          thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]), 
                                          thrust::raw_pointer_cast(&d_A_csr_col_idx[0]), 
                                          thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                                          nnzM,
                                          thrust::raw_pointer_cast(&d_M_csr_row_pointers[0]), 
                                          thrust::raw_pointer_cast(&d_M_csr_col_idx[0]), 
                                          thrust::raw_pointer_cast(&d_M_csr_vals[0]),
                                          thrust::raw_pointer_cast(&d_P[0]),
                                          thrust::raw_pointer_cast(&d_Q[0]),
                                          infoM));

  HIP_CHECK(hipDeviceSynchronize());
  // this operates in such a way that it over-writes rhs (b) with the solution. So we init x with with contents of rhs.
  thrust::copy(thrust::device, h_b.begin(), h_b.end(), d_x.begin() );

  HIP_CHECK(hipDeviceSynchronize());
  ROCBLAS_CHECK(rocsolver_dcsrrf_solve(handle_rocblas,
                                       nn,
                                       1,
                                       nnzM,
                                       thrust::raw_pointer_cast(&d_M_csr_row_pointers[0]), 
                                       thrust::raw_pointer_cast(&d_M_csr_col_idx[0]), 
                                       thrust::raw_pointer_cast(&d_M_csr_vals[0]),
                                       thrust::raw_pointer_cast(&d_P[0]),
                                       thrust::raw_pointer_cast(&d_Q[0]),
                                       thrust::raw_pointer_cast(&d_x[0]),
                                       nn,
                                       infoM)); 

  HIP_CHECK(hipDeviceSynchronize());

  backward_error_estimate(nn,
                          thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]), 
                          thrust::raw_pointer_cast(&d_A_csr_col_idx[0]), 
                          thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                          thrust::raw_pointer_cast(&d_x[0]),
                          thrust::raw_pointer_cast(&d_b[0]),
                          thrust::raw_pointer_cast(&d_res[0]));
  HIP_CHECK(hipDeviceSynchronize());

  // now we need vector d_res inf norm 
  nrm_it = thrust::max_element(d_res.begin(), d_res.end(), abs_compare());

  omega = *nrm_it; 
  printf("Second system: rocSolverRf backward error (computed through hip, smaller value is better): %16.16e\n\n", omega);

  // Third matrix

  // Read new A and new rhs
  if (matrixFileName3 == NULL) {
    printf("\n");
    printf("Third matrix NOT provided, exiting ... \n");
    return -1;
  }

  // You do not have to do it in an actual application, but the matrix reading routine ALLOCs these again so...
  free(h_A_coo_rows);
  free(h_A_coo_cols);
  free(h_A_coo_vals);

  printf("\n\nReading third MTX file ... \n");

  read_mtx_file_into_coo(matrixFileName3, &nn, &nm, &nnnz, &h_A_coo_rows, &h_A_coo_cols, &h_A_coo_vals);

  if (rhsFileName3 != NULL) {
    printf("Reading RHS file ... \n");
    read_rhs_file(rhsFileName3, thrust::raw_pointer_cast(&h_b[0]));
  } else {
    printf("Creating RHS ...\n");
    thrust::fill(thrust::host, h_b.begin(), h_b.end(), 1.0);
  }

  thrust::copy(thrust::device, h_b.begin(), h_b.end(), d_b.begin() );
  
  printf("\n\n============================== \n\n");
  printf("Third matrix info:\n");
  printf("\t Name:                                 %s\n", matrixFileName3);
  printf("\t Size:                                 %d x %d\n", nn, nn);
  printf("\t Nnz (expanded):                       %d\n", nnnz);
  if (rhsFileName3 != NULL) {
    printf("\t Rhs present?                          YES\n");
    printf("\t Rhs file:                             %s\n", rhsFileName3);
  } else {
    printf("\t Rhs present?                          NO\n");
  }
  printf("\n\n============================== \n\n");

  // COO 2 CSC
  status = umfpack_di_triplet_to_col(nn,
                                     nm,
                                     nnnz,
                                     h_A_coo_rows,
                                     h_A_coo_cols,
                                     h_A_coo_vals,
                                     thrust::raw_pointer_cast(&h_A_csc_col_pointers[0]),
                                     thrust::raw_pointer_cast(&h_A_csc_row_idx[0]),
                                     thrust::raw_pointer_cast(&h_A_csc_vals[0]), 
                                     NULL);
  // status == 0 when thigs are OK
  if (status != 0)  {
    printf("COO 2 CSC returned status %d \n", status);
  }

  // update on the GPU
  thrust::copy(thrust::device, h_A_csc_col_pointers.begin(), h_A_csc_col_pointers.end(), d_A_csc_col_pointers.begin() );
  thrust::copy(thrust::device, h_A_csc_row_idx.begin(), h_A_csc_row_idx.end(), d_A_csc_row_idx.begin() );
  thrust::copy(thrust::device, h_A_csc_vals.begin(), h_A_csc_vals.end(), d_A_csc_vals.begin() );


  // NOW CSC 2 CSR

  HIP_CHECK(hipDeviceSynchronize());
  ROCSPARSE_CHECK(rocsparse_dcsr2csc(handle_rocsparse,
                                     nn,
                                     nn,
                                     nnnz,
                                     thrust::raw_pointer_cast(&d_A_csc_vals[0]),
                                     thrust::raw_pointer_cast(&d_A_csc_col_pointers[0]),
                                     thrust::raw_pointer_cast(&d_A_csc_row_idx[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_col_idx[0]),
                                     thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]),
                                     rocsparse_action_numeric,
                                     rocsparse_index_base_zero,
                                     d_csc2csr_buffer));

  HIP_CHECK(hipDeviceSynchronize());

  ROCBLAS_CHECK(rocsolver_dcsrrf_refactlu(handle_rocblas,
                                          nn,
                                          nnnz,
                                          thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]), 
                                          thrust::raw_pointer_cast(&d_A_csr_col_idx[0]), 
                                          thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                                          nnzM,
                                          thrust::raw_pointer_cast(&d_M_csr_row_pointers[0]), 
                                          thrust::raw_pointer_cast(&d_M_csr_col_idx[0]), 
                                          thrust::raw_pointer_cast(&d_M_csr_vals[0]),
                                          thrust::raw_pointer_cast(&d_P[0]),
                                          thrust::raw_pointer_cast(&d_Q[0]),
                                          infoM));

  HIP_CHECK(hipDeviceSynchronize());
  // this operates in such a way that it over-writes rhs (b) with the solution. So we init x with with contents of rhs.
  thrust::copy(thrust::device, h_b.begin(), h_b.end(), d_x.begin() );

  HIP_CHECK(hipDeviceSynchronize());
  ROCBLAS_CHECK(rocsolver_dcsrrf_solve(handle_rocblas,
                                       nn,
                                       1,
                                       nnzM,
                                       thrust::raw_pointer_cast(&d_M_csr_row_pointers[0]), 
                                       thrust::raw_pointer_cast(&d_M_csr_col_idx[0]), 
                                       thrust::raw_pointer_cast(&d_M_csr_vals[0]),
                                       thrust::raw_pointer_cast(&d_P[0]),
                                       thrust::raw_pointer_cast(&d_Q[0]),
                                       thrust::raw_pointer_cast(&d_x[0]),
                                       nn,
                                       infoM)); 

  HIP_CHECK(hipDeviceSynchronize());

  backward_error_estimate(nn,
                          thrust::raw_pointer_cast(&d_A_csr_row_pointers[0]), 
                          thrust::raw_pointer_cast(&d_A_csr_col_idx[0]), 
                          thrust::raw_pointer_cast(&d_A_csr_vals[0]),
                          thrust::raw_pointer_cast(&d_x[0]),
                          thrust::raw_pointer_cast(&d_b[0]),
                          thrust::raw_pointer_cast(&d_res[0]));
  HIP_CHECK(hipDeviceSynchronize());

  // now we need vector d_res inf norm 
  nrm_it = thrust::max_element(d_res.begin(), d_res.end(), abs_compare());

  omega = *nrm_it; 
  printf("Third system: rocSolverRf backward error (computed through hip, smaller value is better): %16.16e\n\n", omega);



  klu_free_symbolic(&Symbolic, &Common);
  klu_free_numeric(&Numeric, &Common); 
  //free cpu variables; thrust has a garbage collector 

  free(h_A_coo_rows);
  free(h_A_coo_cols);
  free(h_A_coo_vals);
  return 0 ;
}


