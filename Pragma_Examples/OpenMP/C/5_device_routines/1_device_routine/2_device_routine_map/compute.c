#pragma omp declare target
void compute(double *x){
   *x = 1.0;
}
