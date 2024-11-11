
extern double *constants;

void compute(int cindex, double *x){
   *x = 1.0 + constants[cindex];
}

