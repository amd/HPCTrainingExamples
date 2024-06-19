#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <immintrin.h>

typedef double v4dd __attribute__ ((vector_size (32)));

int main()
  {
    long i,N=10000000000,j,s=2,n,VLEN=4,m;
    long name_length=0, start_index, end_index, unroll=VLEN, inf=N;
    double sum=0.0;

    v4dd  __D_POW ;
    v4dd  __D_N   ;

    /* compute start_index and end_index point */
    start_index = N - (N % unroll);
    end_index = unroll;

    /* pre-vector loop */
    for(n=inf-1;n>start_index;n--) {
       sum += pow(1.0/(double)n,s);
    }

    // __P_SUM[0 ... VLEN] = 0 */
    v4dd __P_SUM   = {0.0,0.0,0.0,0.0};

     // __ONE[0 ... VLEN] = 1
    v4dd __ONE     = {1.0,1.0,1.0,1.0};

    v4dd __inc     = {(double)VLEN,(double)VLEN,(double)VLEN,(double)VLEN};

    // __DEC[0 ... VLEN] = VLEN
    v4dd __DEC  = __inc, __L , __INV  ;

    for(i=0;i<VLEN;i++) {
       __L[i]		= (double)(inf - 1 - i);
    }

  	for (n=start_index; n > end_index ; n -= unroll ) {
        __D_POW 	    =   __L * __L;
        __L 			    -=  __DEC;
        __INV		      =   __ONE / __D_POW;
        __P_SUM	   	  +=  __INV;
    }

    /* reduce partial sums to final sum */
    for(i=0;i<VLEN;i++) {
       sum += __P_SUM[i];
    }

    /* post-vector loop index cleanup, serial section */
    for(n=end_index-1;n>=1;n--) {
       sum += pow(1.0/(double)n,2.0);
    }

    printf("ζ(%li) = %18.16f\n",s,sum);
    if (s == 2) {
      double pi = sqrt(6.0 * sum);
      printf(" π = %18.16f\n",pi);
    }
   return(0);
  }
