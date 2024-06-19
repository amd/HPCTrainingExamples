#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define true  1
#define false 0

long atol(const char *nptr);

/* sharded zeta */
double zeta(long s, long Nstart, long Nstop) {
  double _zeta;
  if (s != 2) { 
     for(long n=Nstop;n>=Nstart;n--) _zeta += pow((double)n,-s); 
    }
  else { 
    for(long n=Nstop;n>=Nstart;n--) _zeta += (1.0/(double)n)*(1.0/(double)n); }
  return(_zeta);
}

int main(int argc, char **argv)
{
  double _z = 0.0 ,pso6 = 0.0, _t;
  long i, N = 10000000000, s = 2, _N, _s;
  char c;
  int thr, Nthr, debug = false, timing = false;

  long Nstart, Nstop, N_per;

  /* start parallel execution immediately */
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &thr);
  MPI_Comm_size(MPI_COMM_WORLD, &Nthr);
  MPI_Status stat;


  /* 
    parse command line options:
        -n long_int for number of elements of sum, defaults to 10 billion 
        -s argument of Riemann Zeta Function, defaults to 2

  */
  while ((c = getopt(argc, argv, "n:s:dt")) != -1) {
    switch (c) {
      case 'n':
        _N = atol((char *)optarg);
        if ((_N > 0) && (_N < LONG_MAX)) N = _N;
        break;
      case 's':
        _s = atol((char *)optarg);
        if ((_s > 0) && (_s < LONG_MAX)) s = _s;
        break;  
      case 'd':
        debug = true;
        break;
      case 't':
        timing = true;
        break;
    }
  }

  /* 
    actual sum, perform in reverse order to get better 
    roundoff error accumulation.  Specialize for s = 2 
    to remove pow function call
  */

  /* each MPI rank computes its own local versions of these, to avoid communicating them */
  N_per = (long)ceil(N/Nthr);
  Nstart= 1 + (thr-0) * N_per ; 
  Nstop = 0 + (thr+1) * N_per ;

  if (debug) {
    printf("thr = %i\nN, N_per = %li, %li\nNstart, Nstop = %li, %li\n\n",thr,N,N_per,Nstart,Nstop);
  }
  struct timespec start,end;
  clock_gettime(CLOCK_REALTIME, &start);
  _z=zeta( s, Nstart, Nstop);
  clock_gettime(CLOCK_REALTIME, &end);
  _t = (((double)end.tv_sec*1.0e9 + end.tv_nsec) - ((double)start.tv_sec*1e9 + start.tv_nsec))/1.0e9;
  if (timing) {
    printf("thr = %i, loop time (s) = %18.15f\n",thr,_t);
  }

  /* and now the collective operation, a sum reduction primative */

  MPI_Reduce(&_z, &pso6, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  /* now only print from thread/rank 0 */
  if (thr == 0) {
  printf("ζ(%li) = %18.16f\n",s,pso6);

  if (s == 2) {
      double pi = sqrt(6.0 * pso6);
      printf(" π = %18.16f\n",pi);
    }
  }
 MPI_Finalize();

 exit(0);
}
