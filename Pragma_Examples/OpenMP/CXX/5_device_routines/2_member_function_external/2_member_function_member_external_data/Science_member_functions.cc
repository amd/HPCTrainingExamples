#include "Science.hh"

#pragma omp declare target
void Science::compute(double *x, int N){
   *x = Science::init_value;
}
#pragma omp end declare target
