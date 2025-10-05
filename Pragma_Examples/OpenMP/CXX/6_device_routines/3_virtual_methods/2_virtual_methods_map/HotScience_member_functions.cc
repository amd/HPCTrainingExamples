#include "HotScience.hh"

#pragma omp declare target
void HotScience::compute(double *x, int N){
   *x = 5.0;
}
#pragma omp end declare target
