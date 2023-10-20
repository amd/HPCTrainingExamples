/***************************************************************************
 Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef _DGEMM_H_
#define _DGEMM_H_


#include <vector>
#include "matrix.h"


struct dgemm_results{
  std::vector<double> flops;
  std::vector<std::string> time_points;
};


/** Run A*B on gpus
 *
 * @param A input matrix A
 * @param B input matrix B
 * @param inter_count number of iterations to perform dgemm
 * @param rep_count number of times to perform dgemm to compute flops
 * @param device_id GPU device id, indexed at 0, to run on
 * @return estimated terraflops and corresponding local time
 *
 */
dgemm_results
run_dgemm(
   matrixd const& A, matrixd const& B,
   int iter_count, int rep_count, int dev_id);


#endif /* _DGEMM_H_ */
