#pragma omp declare target
void compute(double *x){
   *x = 1.0;
}
#pragma omp end declare target
