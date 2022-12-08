/***************************************************************************
 Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
***************************************************************************/

#ifndef ARGS_H_
#define ARGS_H_
#include <string>
#include <vector>

struct args{
  // Matrix dimensions
  int m = 0;
  int n = 0;
  int k = 0;

  std::vector<int> device_ids;

  // number of times to perform
  int iter_count = 1;

  // number of times to repeat dgemm while measuring time
  int rep_count = 10;

  std::string output_fn;

  static
  bool
  validate(args const& input){
   return (input.m > 1)
     && (input.n > 0)
     && (input.k > 0)
     && (input.iter_count > 0)
     && (input.rep_count > 0);
  }

};

#endif /* ARGS_H_ */
