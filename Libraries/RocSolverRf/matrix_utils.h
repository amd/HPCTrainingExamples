#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <iostream>
#include "hip_utils.h"

void read_rhs_file(const char *rhsFileName, double *rhs);

void read_mtx_file_into_coo(const char *matrixFileName, 
			    int* nn,
			    int* m,
			    int* nnz,
			    int** A_coo_rows, 
			    int** A_coo_cols, 
			    double** A_coo_vals);

void backward_error_estimate(const int N,
			     const int* ia,
			     const int* ja, 
			     const double* a,
			     const double* x,
			     const double* b,
			     double* result); 

double vector_inf_norm(const int n, 
		     const double* input, 
		     double* buffer,
		     double* result);
struct abs_compare
{
    __host__ __device__
    bool operator()(const double& a, const double& b) const
    {
        return std::abs(a) < std::abs(b);
    }
};
