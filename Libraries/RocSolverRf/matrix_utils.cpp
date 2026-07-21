#include "matrix_utils.h"

void read_mtx_file_into_coo(const char *matrixFileName, 
			    int* nn,
			    int* m,
			    int* nnz,
			    int** A_coo_rows, 
			    int** A_coo_cols, 
			    double** A_coo_vals)
{
  // this reads symmetric matrix but expands into full as it goes (important)
  FILE *fpm = fopen(matrixFileName, "r");
  bool symmetric = 0;
  char lineBuffer[256];
  fgets(lineBuffer, sizeof(lineBuffer), fpm);
  // this reads triangular matrix or general matrix  
  int noVals = 0;


  /* For graphs */
  char *s = strstr(lineBuffer, "pattern");
  if (s != NULL) {
    noVals = 1;
  }

  s = strstr(lineBuffer, "symmetric");
  if (s != NULL) {
    symmetric = 1;
  }

  while (lineBuffer[0] == '%') {
    fgets(lineBuffer, sizeof(lineBuffer), fpm);
  }

  // first line (after comments) is size and nnz, need this info to allocate memory
  // if matrix is symmetric, we get the number of values to read, not the true number of non-zeros

  sscanf(lineBuffer, "%d %d %d", nn, m, nnz);
  //allocate
  //*nnz = *nnz * 2 - *nn;

  *A_coo_vals = (double*) calloc(*nnz * 2, sizeof(double));
  *A_coo_rows = (int*) calloc(*nnz * 2, sizeof(int));
  *A_coo_cols = (int*) calloc(*nnz * 2, sizeof(int));

  //read
  int r, c;
  double val;
  int i = 0;
  while (fgets(lineBuffer, sizeof(lineBuffer), fpm) != NULL)
  {

    if (noVals == 0) {
      sscanf(lineBuffer, "%d %d %lf", &r, &c, &val);
    } else {
      sscanf(lineBuffer, "%d %d", &r, &c);
      val = 1.0;
    }
    (*A_coo_rows)[i] = r - 1;
    (*A_coo_cols)[i] = c - 1;
    (*A_coo_vals)[i] = val;
    i++;
    if ((symmetric) && (r != c)) {
      (*A_coo_rows)[i] = c - 1;
      (*A_coo_cols)[i] = r - 1;
      (*A_coo_vals)[i] = val;
      i++;
    }
    if ((c < 1) || (r < 1))
      printf("Problem with file: read %d %d %16.16f \n", r - 1, c - 1, val);
  }
  *nnz = i;
  fclose(fpm);
}

void read_rhs_file(const char *rhsFileName, double *rhs)
{

  FILE* fpr = fopen(rhsFileName, "r");
  char lineBuffer[256];

  fgets(lineBuffer, sizeof(lineBuffer), fpr);
  while (lineBuffer[0] == '%') {
    fgets(lineBuffer, sizeof(lineBuffer), fpr);
  }
  int N, m;
  sscanf(lineBuffer, "%d %d", &N, &m);
  int i = 0;
  double val;
  //allocate

  while (fgets(lineBuffer, sizeof(lineBuffer), fpr) != NULL)
  {
    sscanf(lineBuffer, "%lf", &val);
    rhs[i] = val;
    i++;
  }
  fclose(fpr);
}

// error estimate formula, omega, is given as max_i  (|Ax-b|_i)/ (|A||b|-|x|)_i (interpret 0/0 as 0)
// this can be written way more efficient (see ROCm blog post on SpMV)

__global__ void backward_error_estimate_kernel(const int N,
					       const int* ia,
					       const int* ja, 
					       const double* a,
					       const double* x,
					       const double* b,
					       double* result)
{
  double sum_enum, sum_denom;
  
  auto const row_start = hipThreadIdx_x + hipBlockIdx_x * hipBlockDim_x; 
  auto const row_inc = hipBlockDim_x * hipGridDim_x;    
   
  for( auto irow = row_start; irow < N; irow += row_inc) {
    sum_enum = 0.0;
    sum_denom = 0.0;
    for (int j = ia[irow]; j < ia [irow + 1]; ++j) {
      sum_enum += ( a[j] * x[ja[j]]) ;
      sum_denom += (std::abs(a[j]) * std::abs(x[ja[j]]));
    }
    sum_enum = std::abs(b[irow] - sum_enum);
    sum_denom += std::abs(b[irow]); 
    // avoid zero division
    if (sum_denom != 0.0) {
      result[irow] = sum_enum / sum_denom;
    } else {
      result[irow] = sum_enum;
    } 
  }
}

// Vector inf norm is handled through roctrust; need this custom comparison to be able to use max_element with absolute value
// Reduction handled by rocThrust



void backward_error_estimate(const int N,
			     const int* ia,
			     const int* ja, 
			     const double* a,
			     const double* x,
			     const double* b,
			     double* result) 
{

  backward_error_estimate_kernel<<<N / 1924 + 1, 1024>>>(N,
							 ia,
							 ja, 
							 a,
							 x,
							 b,
							 result); 
}


