#include <stdlib.h>

double *constants;

void initialize_constants(int isize){
   constants = (double *)malloc(isize*sizeof(double));
   for (int i = 0; i< isize; i++) {
      constants[i] = (double)i;
   }
}
