// Copyright (c) 2024 AMD HPC Application Performance Team
// Author: Bob Robey, Bob.Robey@amd.com
// MIT License

#include <stdlib.h>
#include <stdio.h>

void compute(int cindex, double *x);

int main(int argc, char *argv[]){
   int N=1000;
   double *x = (double *)malloc(N*sizeof(double));
#pragma omp target enter data map (alloc:x[0:N])
#pragma omp target teams distribute parallel for
   for (int k = 0; k < N; k++){
      x[k] = 0.0;
   }

#pragma omp target teams distribute parallel for
   for (int k = 0; k < N; k++){
      int cindex = k%10;
      compute(cindex, &x[k]);
   }

   double sum = 0.0;
#pragma omp target teams distribute parallel for reduction(+:sum)
   for (int k = 0; k < N; k++){
      sum += x[k];
   }

   printf("Result: sum of x is %lf\n",sum);

#pragma omp target exit data map (release:x[0:N])
   free(x);
}
      
