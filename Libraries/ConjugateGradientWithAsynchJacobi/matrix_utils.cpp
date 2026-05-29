#include "matrix_utils.h"

void read_mtx_file_into_coo(const char *matrixFileName, 
			    int* nn,
			    int* m,
                            int* nnz_or,
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

  sscanf(lineBuffer, "%d %d %d", nn, m, nnz_or);
  //allocate
  //*nnz = *nnz * 2 - *nn;

  *A_coo_vals = (double*) calloc(*nnz_or * 2, sizeof(double));
  *A_coo_rows = (int*) calloc(*nnz_or * 2, sizeof(int));
  *A_coo_cols = (int*) calloc(*nnz_or * 2, sizeof(int));

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

// note: this works even if matrix is not sorted!
// also, it expands symmetric matrix to full

/* COO to CSR */
void coo_to_csr(int n,
		int nnz_unpacked,
		int* A_coo_rows, 
		int* A_coo_cols, 
		double* A_coo_vals,
		int* A_csr_row_ptr, 
		int* A_csr_col_idx, 
		double* A_csr_vals)

{
  /* first, decide how many nnz we have in each row */
  int* nnz_counts;
  nnz_counts = (int *) calloc(n, sizeof(int));
  for (int i = 0; i < nnz_unpacked; ++i) {
    nnz_counts[A_coo_rows[i]]++;
  }

  /* allocate full CSR structure */

  indexPlusValue *tmp = (indexPlusValue*) calloc(nnz_unpacked, sizeof(indexPlusValue));

  /* create row pointers */
  A_csr_row_ptr[0] = 0;
  for (int i = 1; i < n + 1; ++i) {
    A_csr_row_ptr[i] = A_csr_row_ptr[i - 1] + nnz_counts[i - 1];
  }

  int *nnz_shifts = (int *) calloc(n, sizeof(int));
  int r, start;

  for (int i = 0; i < nnz_unpacked; ++i) {
    /* which row 8 */ 
    r = A_coo_rows[i];
    start = A_csr_row_ptr[r];
    if ((start + nnz_shifts[r]) > nnz_unpacked) {
      printf("index out of bounds\n");
    }

    tmp[start + nnz_shifts[r]].idx = A_coo_cols[i];
    tmp[start + nnz_shifts[r]].value = (double) A_coo_vals[i];

    nnz_shifts[r]++;

  }
  /* now sort whatever is inside rows */

  for (int i = 0; i < n; ++i)
  {

    //now sorting (and adding 1)
    int colStart = A_csr_row_ptr[i];
    int colEnd = A_csr_row_ptr[i + 1];
    int length = colEnd - colStart;

    qsort(&tmp[colStart], length, sizeof(indexPlusValue), indexPlusValue_comp);
  }

  //and copy
  for (int i = 0; i < nnz_unpacked; ++i)
  {
    A_csr_col_idx[i] = tmp[i].idx;
    A_csr_vals[i] = tmp[i].value;
  }
#if 0
  for (int i = 0; i < 10; i++) {
    printf("this is row %d \n", i);
    for (int j = A_csr_row_ptr[i]; j < A_csr_row_ptr[i + 1]; ++j) { 
      printf("  (%d, %f)  ", A_csr_col_idx[j], A_csr_vals[j] );      

    }
    printf("\n");
  }
#endif
  free(nnz_counts);
  free(tmp);
  free(nnz_shifts);
}

