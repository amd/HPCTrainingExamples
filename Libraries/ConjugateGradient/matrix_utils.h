#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <iostream>
#include "hip_utils.h"

/* needed for easy sorting in format conversion function*/

struct indexPlusValue
{
  double value;
  int idx;
};

typedef struct indexPlusValue indexPlusValue;

/* neded for qsort */
static int indexPlusValue_comp(const void *a, const void *b)
{
  const struct indexPlusValue *da = (indexPlusValue *)a;
  const struct indexPlusValue *db = (indexPlusValue *)b;

  return da->idx < db->idx ? -1 : da->idx > db->idx;
}


void read_rhs_file(const char *rhsFileName, double *rhs);

void read_mtx_file_into_coo(const char *matrixFileName, 
			    int* nn,
			    int* m,
			    int* nnz_or,
			    int* nnz,
			    int** A_coo_rows, 
			    int** A_coo_cols, 
			    double** A_coo_vals);

void coo_to_csr(int n,
                int nnz_unpacked, 
		int* A_coo_rows, 
		int* A_coo_cols, 
		double* A_coo_vals,
		int* A_csr_row_ptr, 
		int* A_csr_col_idx, 
		double* A_csr_vals);
