#include "Science.hh"

#pragma omp declare target
void Science::compute(double *x, int N){
   *x = 1.0;
}
#pragma omp end declare target
