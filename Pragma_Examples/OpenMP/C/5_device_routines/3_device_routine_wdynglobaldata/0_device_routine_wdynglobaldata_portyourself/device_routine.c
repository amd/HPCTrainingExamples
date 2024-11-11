// Copyright (c) 2024 AMD HPC Application Performance Team
// Author: Bob Robey, Bob.Robey@amd.com
// MIT License

#include <stdlib.h>
#include <stdio.h>

void initialize_constants(int isize);
void compute(int cindex, double *x);

int main(int argc, char *argv[]){
   int N=1000;
   int isize=10;
   initialize_constants(isize);

   double *x = (double *)malloc(N*sizeof(double));
   for (int k = 0; k < N; k++){
      x[k] = 0.0;
   }

   for (int k = 0; k < N; k++){
      int cindex = k%isize;
      compute(cindex, &x[k]);
   }

   double sum = 0.0;
   for (int k = 0; k < N; k++){
      sum += x[k];
   }

   printf("Result: sum of x is %lf\n",sum);

   free(x);
}
      
