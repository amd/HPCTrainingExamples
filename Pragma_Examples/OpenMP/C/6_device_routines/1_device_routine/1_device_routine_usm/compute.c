#pragma omp requires unified_shared_memory
#pragma omp declare target
void compute(double *x){
   *x = 1.0;
}
#pragma omp end declare target
