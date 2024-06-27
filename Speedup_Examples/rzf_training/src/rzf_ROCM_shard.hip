#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <unistd.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <ctime>
#include <iostream>
#include <rocprim/rocprim.hpp>

#define Nthr 1024

// timing of functions
clock_t start,end;


/* invert and square every element of the input array in parallel */
__global__ void _zeta(double *psum, long base, long Npanels)
{

  long Nstart, Nend, globalIdx, i;
  double r2 = 0.0;

  // sum over panels
  for(i = 0; i<Npanels; i++) {
    globalIdx = blockDim.x * blockIdx.x  + threadIdx.x + 1 + base * i; 
    r2 += (1.0 / globalIdx) * (1.0 / globalIdx);
  }
  atomicAdd(&psum[threadIdx.x], r2);
}

#define true 1
#define false 0

long atol(const char *nptr);

int main(int argc, char **argv)
{
    long Nblocks, j, Npanels = 1;
    long i, N = 10000000000, s = 2, _N, _s;
    char c;
    int thr, debug = false, timing = false;
    double sum = 0.0;


    while ((c = getopt(argc, argv, "n:s:dt")) != -1)
    {
      switch (c)
      {
      case 'n':
        _N = atol((char *)optarg);
        if ((_N > 0) && (_N < LONG_MAX))
          N = _N;
        break;
      case 's':
        _s = atol((char *)optarg);
        if ((_s > 0) && (_s < LONG_MAX))
          s = _s;
        break;
      case 'd':
        debug = true;
        break;
      case 't':
        timing = true;
        break;
      }
    }
    _N = N - (N %  Nthr);
    
    Nblocks = (long)ceil((double)_N/(double)Nthr);
    if (Nblocks > 1024) {
      Nblocks = 1024;
      Npanels = (long)ceil((double)_N / ((double)Nblocks*Nthr));
    }
    
    double *ps_h, *ps_d;
    start = std::clock();
    
    ps_h = (double*)calloc(Nthr,sizeof(double));
    for(int i=0;i<Nthr;i++) ps_h[i] = 0.0;

    hipMalloc(&ps_d, Nthr * sizeof(double));
    hipMemcpy(ps_d, ps_h, Nthr * sizeof(double), hipMemcpyHostToDevice);

    // we will handle the remaining portions on CPU, here
    for(i = N ; i>_N; i--) {
      sum += pow(1.0/(double)i,2.0);
    }

    if (debug) {
      printf("initial psum = %18.15f\n",sum);
      printf("Npanels = %li\nNblocks = %li\n",Npanels,Nblocks);
      printf("starting GPU kernel\n");
      }

    long base = Nblocks * Nthr ; 
    _zeta<<<Nblocks, Nthr >>>(ps_d, base, Npanels);
    

    if (debug)
    {      
      printf("ending GPU kernel\n");
    }

    hipMemcpy(ps_h,ps_d,Nblocks * sizeof(double), hipMemcpyDeviceToHost);
    for(i = 0 ; i<Nthr; i++) {
      sum += ps_h[i];
    }

    end = std::clock();


    // Clean up
    hipFree(ps_d);
    free(ps_h);
    double pi = sqrt(6.0 * sum);
    printf("ζ(2) = %18.16f\n",sum);
    printf(" π = %18.16f\n",pi);
}
