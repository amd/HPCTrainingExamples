#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>
#include <stdlib.h>

double zeta(long s, long N) {
  double _zeta;
  if (s != 2) { for(long n=N;n>=1;n--) _zeta += pow((double)n,-s); }
  else { for(long n=N;n>=1;n--) _zeta += (1.0/(double)n)*(1.0/(double)n); }
  return _zeta;
}

void main(int argc, char **argv)
{
  double _z;
  long i, N = 10000000000, s = 2;
  char c;

  /* 
    parse command line options:
        -n long_int for number of elements of sum, defaults to 10 billion 
        -s argument of Riemann Zeta Function, defaults to 2

  */
  while ((c = getopt(argc, argv, "n:s:")) != -1) {
    switch (c) {
      case 'n':
        long _N = (long)atol((char *)optarg);
        if ((_N > 0) && (_N < LONG_MAX)) N = _N;
        break;
      case 's':
        long _s = (long)atol((char *)optarg);
        if ((_s > 0) && (_s < LONG_MAX)) s = _s;
        break;  
    }
  }

  /* 
    actual sum, perform in reverse order to get better 
    roundoff error accumulation.  Specialize for s = 2 
    to remove pow function call
  */
  
  _z = zeta(s,N);
  printf("ζ(%li) = %18.16f\n",s,_z);

  if (s == 2) {
    double pi = sqrt(6.0 * _z);
    printf(" π = %18.16f\n",pi);
  }
 exit(0);
}
