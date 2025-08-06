#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#include "malloc2D.h"
#include "timer.h"

#define SWAP_PTR(xnew,xold,xtmp) (xtmp=xnew, xnew=xold, xold=xtmp)
void parse_input_args(int argc, char **argv, int &jmax, int &imax, int &nprocy, int &nprocx, int &nhalo, int &corners, int &maxIter, int &do_timing, int &do_print);
void Cartesian_print(double **x, int jmax, int imax, int nhalo, int nprocy, int nprocx);
void boundarycondition_update(double **x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop);
void ghostcell_update(double **x, int nhalo, int corners, int jsize, int isize,
      int nleft, int nrght, int nbot, int ntop, int do_timing);
void haloupdate_test(int nhalo, int corners, int jsize, int isize, int nleft, int nrght, int nbot, int ntop,
      int jmax, int imax, int nprocy, int nprocx, int do_timing, int do_print);

double boundarycondition_time=0.0, ghostcell_time=0.0;

// this version computes the solution first on the cells that
// do not need to use information from the halo cells, then
// performs a halo exchange, and finally advances the cells
// that use information from the halo.
// this version serves as an initial step towards having
// an overlap of GPU kernel compute with CPU MPI exchange
// (see Ver2 in HIP version of Ghost Exchange code)

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (rank == 0) printf("------> Initializing the Problem\n");

   int imax = 2000, jmax = 2000;
   int nprocx = 0, nprocy = 0;
   int nhalo = 2, corners = 0;
   int do_timing = 0;
   int do_print = 0;
   int maxIter = 1000;

   parse_input_args(argc, argv, jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing, do_print);
 
   struct timespec tstart_stencil, tstart_total;
   double stencil_time=0.0, total_time;
   cpu_timer_start(&tstart_total);

   int xcoord = rank%nprocx;
   int ycoord = rank/nprocx;

   int nleft = (xcoord > 0       ) ? rank - 1      : MPI_PROC_NULL;
   int nrght = (xcoord < nprocx-1) ? rank + 1      : MPI_PROC_NULL;
   int nbot  = (ycoord > 0       ) ? rank - nprocx : MPI_PROC_NULL;
   int ntop  = (ycoord < nprocy-1) ? rank + nprocx : MPI_PROC_NULL;

   int ibegin = imax *(xcoord  )/nprocx;
   int iend   = imax *(xcoord+1)/nprocx;
   int isize  = iend - ibegin;
   int jbegin = jmax *(ycoord  )/nprocy;
   int jend   = jmax *(ycoord+1)/nprocy;
   int jsize  = jend - jbegin;

   /* The halo update both updates the ghost cells and the boundary halo cells. To be precise with terminology,
    * the ghost cells only exist for multi-processor runs with MPI. The boundary halo cells are to set boundary
    * conditions. Halos refer to both the ghost cells and the boundary halo cells.
    */
   haloupdate_test(nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, jmax, imax, nprocy, nprocx, do_timing, do_print);

   double** xtmp;
   // This offsets the array addressing so that the real part of the array is from 0,0 to jsize,isize
   double** x    = malloc2D(jsize+2*nhalo, isize+2*nhalo, nhalo, nhalo);
   double** xnew = malloc2D(jsize+2*nhalo, isize+2*nhalo, nhalo, nhalo);

   if (! corners) { // need to initialize when not doing corners so there is no uninitialized memory
      for (int j = -nhalo; j < jsize+nhalo; j++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            x[j][i] = 0.0;
         }
      }
   }

   for (int j = 0; j < jsize; j++){
      for (int i = 0; i < isize; i++){
         x[j][i] = static_cast<double>(rank) + 5.0;
      }
   }

   int ispan=5, jspan=5;
   if (ispan > imax/2) ispan = imax/2;
   if (jspan > jmax/2) jspan = jmax/2;
   for (int j = jmax/2 - jspan; j < jmax/2 + jspan; j++){
      for (int i = imax/2 - ispan; i < imax/2 + ispan; i++){
         if (j >= jbegin && j < jend && i >= ibegin && i < iend) {
            x[j-jbegin][i-ibegin] = 400.0 + 0.1 * (double)(rank + j);
         }
      }
   }

   boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);

   if (do_print == 1) {
      if (rank == 0) printf("------> Initial State \n");
      // note: this ghostcell_update is only for printing purposes to show that the
      // solution matches the one produced by Orig also on the halo for debugging
      // the value of the solution on the halo does not matter as long as
      // it is up to date when it is used to compute the solution on cells that need it
      ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);
      Cartesian_print(x, jmax, imax, nhalo, nprocy, nprocx);
   }

   printf("------> Advancing the Solution\n");
   for (int iter = 0; iter < maxIter; iter++){

      cpu_timer_start(&tstart_stencil);
      // advance soln on cells that don't need halo information
      // note: we assume that the halo size is greater than or equal
      // to the the operator's stencil reach (which for the blur is 1) 
      for (int j = nhalo; j < jsize - nhalo; j++){
         for (int i = nhalo; i < isize - nhalo; i++){
            xnew[j][i] = ( x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i] )/5.0;
         }
      }
      stencil_time += cpu_timer_stop(tstart_stencil);

      ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);

      cpu_timer_start(&tstart_stencil);
     // now we advance the soln on those cells that need the halo

     // left
     for (int j = 0; j < jsize; j++){
        for (int i = 0; i < nhalo; i++){
           xnew[j][i] = ( x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i] )/5.0;
        }
     }

     // right
     for (int j = 0; j < jsize; j++){
        for (int i = isize - nhalo; i < isize; i++){
           xnew[j][i] = ( x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i] )/5.0;
        }
     }

     //top
     for (int j = jsize - nhalo; j < jsize; j++){
        for (int i = nhalo; i < isize - nhalo; i++){
           xnew[j][i] = ( x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i] )/5.0;
        }
     }

     //bottom
     for (int j = 0; j < nhalo; j++){
        for (int i = nhalo; i < isize - nhalo; i++){
           xnew[j][i] = ( x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i] )/5.0;
        }
     }
     stencil_time += cpu_timer_stop(tstart_stencil);

    boundarycondition_update(xnew, nhalo, jsize, isize, nleft, nrght, nbot, ntop);

    if (do_print == 1) {
       // halo update on soln
       // NOTE: this is just for visualziation
       // we do not care in general what the solution is in the halo
       // as long as it is correct when we need it to compute the next
       // value of the solution
       ghostcell_update(xnew, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);
    }
      
    cpu_timer_start(&tstart_stencil);
    SWAP_PTR(xnew, x, xtmp);
    stencil_time += cpu_timer_stop(tstart_stencil);

    if (iter%10 == 0 && rank == 0) printf("        Iter %d\n",iter);
    if (do_print == 1) {
       Cartesian_print(x, jmax, imax, nhalo, nprocy, nprocx);
    }
   }
   total_time = cpu_timer_stop(tstart_total);

   if (rank == 0){
      printf("------> Printing Timings\n");
      printf("        Solution Advancement: %f \n", stencil_time);
      printf("        Boundary Condition Enforcement:  %f \n", boundarycondition_time);
      printf("        Ghost Cell Update:  %lf \n", ghostcell_time);
      printf("        Total: %f\n",total_time);
   }

   malloc2D_free(x, nhalo);
   malloc2D_free(xnew, nhalo);

   MPI_Finalize();
   exit(0);
}

void boundarycondition_update(double **x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop)
{
   struct timespec tstart_boundarycondition;
   cpu_timer_start(&tstart_boundarycondition);

// Boundary conditions -- constant
   if (nleft == MPI_PROC_NULL){
      for (int j = 0; j < jsize; j++){
         for (int ll=-nhalo; ll<0; ll++){
            x[j][ll] = x[j][0];
         }
      }
   }

   if (nrght == MPI_PROC_NULL){
      for (int j = 0; j < jsize; j++){
         for (int ll=0; ll<nhalo; ll++){
            x[j][isize+ll] = x[j][isize-1];
         }
      }
   }

   if (nbot == MPI_PROC_NULL){
      for (int ll=-nhalo; ll<0; ll++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            x[ll][i] = x[0][i];
         }
      }
   }
      
   if (ntop == MPI_PROC_NULL){
      for (int ll=0; ll<nhalo; ll++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            x[jsize+ll][i] = x[jsize-1][i];
         }
      }
   }

   boundarycondition_time += cpu_timer_stop(tstart_boundarycondition);
}

void ghostcell_update(double **x, int nhalo, int corners, int jsize, int isize, int nleft, int nrght, int nbot, int ntop, int do_timing)
{
   //if (do_timing) MPI_Barrier(MPI_COMM_WORLD);

   struct timespec tstart_ghostcell;
   cpu_timer_start(&tstart_ghostcell);

   MPI_Request request[4*nhalo];
   MPI_Status status[4*nhalo];

   int jlow=0, jhgh=jsize;
   if (corners) {
      if (nbot == MPI_PROC_NULL) jlow = -nhalo;
      if (ntop == MPI_PROC_NULL) jhgh = jsize+nhalo;
   }
   int jnum = jhgh-jlow;
   int bufcount = jnum*nhalo;

   double xbuf_left_send[bufcount];
   double xbuf_rght_send[bufcount];
   double xbuf_rght_recv[bufcount];
   double xbuf_left_recv[bufcount];
   int offset = 0;

   if (nleft != MPI_PROC_NULL){
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xbuf_left_send[offset] = x[j][ll];
         }
      }
   }
   if (nrght != MPI_PROC_NULL){
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xbuf_rght_send[offset] = x[j][isize-nhalo+ll];
         }
      }
   }

   MPI_Irecv(&xbuf_rght_recv, bufcount, MPI_DOUBLE, nrght, 1001,
             MPI_COMM_WORLD, &request[0]);
   MPI_Isend(&xbuf_left_send, bufcount, MPI_DOUBLE, nleft, 1001,
             MPI_COMM_WORLD, &request[1]);

   MPI_Irecv(&xbuf_left_recv, bufcount, MPI_DOUBLE, nleft, 1002,
             MPI_COMM_WORLD, &request[2]);
   MPI_Isend(&xbuf_rght_send, bufcount, MPI_DOUBLE, nrght, 1002,
             MPI_COMM_WORLD, &request[3]);
   MPI_Waitall(4, request, status);

   if (nrght != MPI_PROC_NULL){
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            x[j][isize+ll] = xbuf_rght_recv[offset];
         }
      }
   }
   if (nleft != MPI_PROC_NULL){
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            x[j][-nhalo+ll] = xbuf_left_recv[offset];
         }
      }
   }

   if (corners) {
      bufcount = nhalo*(isize+2*nhalo);
      MPI_Irecv(&x[jsize][-nhalo],   bufcount, MPI_DOUBLE, ntop, 1001, MPI_COMM_WORLD, &request[0]);
      MPI_Isend(&x[0    ][-nhalo],   bufcount, MPI_DOUBLE, nbot, 1001, MPI_COMM_WORLD, &request[1]);

      MPI_Irecv(&x[     -nhalo][-nhalo], bufcount, MPI_DOUBLE, nbot, 1002, MPI_COMM_WORLD, &request[2]);
      MPI_Isend(&x[jsize-nhalo][-nhalo], bufcount, MPI_DOUBLE, ntop, 1002, MPI_COMM_WORLD, &request[3]);
      MPI_Waitall(4, request, status);
   } else {
      for (int j = 0; j<nhalo; j++){
         MPI_Irecv(&x[jsize+j][0],   isize, MPI_DOUBLE, ntop, 1001+j*2, MPI_COMM_WORLD, &request[0+j*4]);
         MPI_Isend(&x[0+j    ][0],   isize, MPI_DOUBLE, nbot, 1001+j*2, MPI_COMM_WORLD, &request[1+j*4]);

         MPI_Irecv(&x[     -nhalo+j][0], isize, MPI_DOUBLE, nbot, 1002+j*2, MPI_COMM_WORLD, &request[2+j*4]);
         MPI_Isend(&x[jsize-nhalo+j][0], isize, MPI_DOUBLE, ntop, 1002+j*2, MPI_COMM_WORLD, &request[3+j*4]);
         }
      MPI_Waitall(4*nhalo, request, status);
   }

   //if (do_timing) MPI_Barrier(MPI_COMM_WORLD);

   ghostcell_time += cpu_timer_stop(tstart_ghostcell);
}

void haloupdate_test(int nhalo, int corners, int jsize, int isize, int nleft, int nrght, int nbot, int ntop,
      int jmax, int imax, int nprocy, int nprocx, int do_timing, int do_print)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   double** x = malloc2D(jsize+2*nhalo, isize+2*nhalo, nhalo, nhalo);

   if (jsize > 12 || isize > 12) return;

   for (int j = -nhalo; j < jsize+nhalo; j++){
      for (int i = -nhalo; i < isize+nhalo; i++){
         x[j][i] = 0.0;
      }
   }

   for (int j = 0; j < jsize; j++){
      for (int i = 0; i < isize; i++){
         x[j][i] = rank * 1000 + j*10 + i;
      }
   }

   boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
   ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);

   if (do_print == 1) {
      Cartesian_print(x, jmax, imax, nhalo, nprocy, nprocx);
   }

   malloc2D_free(x, nhalo);
}

void parse_input_args(int argc, char **argv, int &jmax, int &imax, int &nprocy, int &nprocx, int &nhalo, int &corners, int &maxIter, int &do_timing, int &do_print)
{
   int c;
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   while ((c = getopt(argc, argv, "cph:I:i:j:tx:y:")) != -1){
      switch(c){
         case 'c':
            corners = 1;
            break;
         case 'h':
            nhalo = atoi(optarg);
            break;
         case 'I':
            maxIter = atoi(optarg);
            break;
         case 'i':
            imax = atoi(optarg);
            break;
         case 'j':
            jmax = atoi(optarg);
            break;
         case 't':
            do_timing = 1;
            break;
         case 'p':
            do_print = 1;
            break;
         case 'x':
            nprocx = atoi(optarg);
            break;
         case 'y':
            nprocy = atoi(optarg);
            break;
         case '?':
            if (optopt == 'h' || optopt == 'I' || optopt == 'i' || optopt == 'j' || optopt == 'x' || optopt == 'y'){
               if (rank == 0) fprintf (stderr, "Option -%c requires an argument.\n", optopt);
               MPI_Finalize();
               exit(1);
            }
            break;
         default:
            if (rank == 0) fprintf(stderr,"Unknown option %c\n",c);
            MPI_Finalize();
            exit(1);
      }
   }

   int xcoord = rank%nprocx;
   int ycoord = rank/nprocx;

   int ibegin = imax *(xcoord  )/nprocx;
   int iend   = imax *(xcoord+1)/nprocx;
   int isize  = iend - ibegin;
   int jbegin = jmax *(ycoord  )/nprocy;
   int jend   = jmax *(ycoord+1)/nprocy;
   int jsize  = jend - jbegin;

   int ierr = 0, ierr_global;
   if (isize < nhalo || jsize < nhalo) {
      if (rank == 0) printf("Error -- local size of grid is less than the halo size\n");
      ierr = 1;
   }
   MPI_Allreduce(&ierr, &ierr_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   if (ierr_global > 0) {
      MPI_Finalize();
      exit(0);
   }
}

void Cartesian_print(double **x, int jmax, int imax, int nhalo, int nprocy, int nprocx)
{
   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   int isize_total=0;
   int isizes[nprocx];
   for (int ii = 0; ii < nprocx; ii++){
      int xcoord = ii%nprocx;
      int ycoord = ii/nprocx;
      int ibegin = imax *(xcoord  )/nprocx;
      int iend   = imax *(xcoord+1)/nprocx;
      isizes[ii] = iend-ibegin;
      isize_total += isizes[ii] + 2*nhalo;
   }

   if (isize_total > 100) return;

   if (rank == 0) {
      printf("     ");
      for (int ii = 0; ii < nprocx; ii++){
         for (int i = -nhalo; i < isizes[ii]+nhalo; i++){
            printf("%6d   ",i);
         }
         printf("   ");
      }
      printf("\n");
   }

   int xcoord = rank%nprocx;
   int ycoord = rank/nprocx;

   int ibegin = imax *(xcoord  )/nprocx;
   int iend   = imax *(xcoord+1)/nprocx;
   int isize  = iend - ibegin;
   int jbegin = jmax *(ycoord  )/nprocy;
   int jend   = jmax *(ycoord+1)/nprocy;
   int jsize  = jend - jbegin;

   double *xrow = (double *)malloc(isize_total*sizeof(double));
   for (int jj=nprocy-1; jj >= 0; jj--){
      int ilen = 0;
      int jlen = 0;
      int jlen_max;
      int *ilen_global = (int *)malloc(nprocs*sizeof(int));
      int *ilen_displ = (int *)malloc(nprocs*sizeof(int));
      if (ycoord == jj) {
         ilen = isize + 2*nhalo;
         jlen = jsize;
      }
      MPI_Allgather(&ilen,1,MPI_INT,ilen_global,1,MPI_INT,MPI_COMM_WORLD);
      MPI_Allreduce(&jlen,&jlen_max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      ilen_displ[0] = 0;
      for (int i=1; i<nprocs; i++){
         ilen_displ[i] = ilen_displ[i-1] + ilen_global[i-1];
      }
      for (int j=jlen_max+nhalo-1; j>=-nhalo; j--){
         MPI_Gatherv(&x[j][-nhalo],ilen,MPI_DOUBLE,xrow,ilen_global,ilen_displ,MPI_DOUBLE,0,MPI_COMM_WORLD);
         if (rank == 0) {
            printf("%3d:",j);
            for (int ii = 0; ii < nprocx; ii++){
               for (int i = 0; i< isizes[ii]+2*nhalo; i++){
                  printf("%8.1lf ",xrow[i+ii*(isizes[ii]+2*nhalo)]);
               }
               printf("   ");
            }
            printf("\n");
         }
      }
      free(ilen_global);
      free(ilen_displ);
      if (rank == 0) printf("\n");
   }
   free(xrow);
}
