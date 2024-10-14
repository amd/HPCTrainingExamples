#include <stdlib.h>

#pragma omp declare target 
double *constants;
#pragma omp end declare target 

void initialize_constants(int isize){
   constants = (double *)malloc(isize*sizeof(double));
#pragma omp target enter data map(alloc:constants[0:isize])
#pragma omp target teams distribute parallel for simd
   for (int i = 0; i< isize; i++) {
      constants[i] = (double)i;
   }
}
