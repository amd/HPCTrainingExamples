#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>

#include "malloc3D.h"
#include "timer.h"

struct neighs
{
   int left; int rght;
   int bot; int top;
   int frnt; int back;
};

struct sizes
{
   int i;
   int j;
   int k;
};
struct procs
{
   int x;
   int y;
   int z;
};

#define SWAP_PTR(xnew,xold,xtmp) (xtmp=xnew, xnew=xold, xold=xtmp)
void parse_input_args(int argc, char **argv, int &kmax, int &jmax, int &imax, 
                      int &nprocz, int &nprocy, int &nprocx,
                      int &nhalo, int &corners, int &do_timing);
void Cartesian_print(double ***x, int kmax, int jmax, int imax, int nhalo, struct procs nproc);
void boundarycondition_update(double ***x, int nhalo, struct sizes size, struct neighs ngh);
void ghostcell_update(double ***x, int nhalo, int corners, struct sizes size,
      struct neighs ngh, struct procs nproc, int do_timing);
void haloupdate_test(int nhalo, int corners, struct sizes size, struct neighs ngh,
      int kmax, int jmax, int imax, struct procs nproc, int do_timing);

double boundarycondition_time=0.0, ghostcell_time=0.0;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (rank == 0) printf("Parallel run with no threads\n");

   int imax = 200, jmax = 200, kmax = 200;
   int nprocx = 0, nprocy = 0, nprocz = 0;
   int nhalo = 2, corners = 0;
   int do_timing = 0;

   parse_input_args(argc, argv, kmax, jmax, imax, nprocz, nprocy, nprocx, nhalo, corners, do_timing);
 
   struct timespec tstart_stencil, tstart_total;
   double stencil_time=0.0, total_time;
   cpu_timer_start(&tstart_total);

   int xcoord = rank%nprocx;
   int ycoord = rank/nprocx%nprocy;
   int zcoord = rank/(nprocx*nprocy);
   //printf("%d:DEBUG -- xcoord %d ycoord %d zcoord %d\n",rank,xcoord,ycoord,zcoord);
   
   int nleft = (xcoord > 0       ) ? rank - 1      : MPI_PROC_NULL;
   int nrght = (xcoord < nprocx-1) ? rank + 1      : MPI_PROC_NULL;
   int nbot  = (ycoord > 0       ) ? rank - nprocx : MPI_PROC_NULL;
   int ntop  = (ycoord < nprocy-1) ? rank + nprocx : MPI_PROC_NULL;
   int nfrnt = (zcoord > 0       ) ? rank - nprocx * nprocy : MPI_PROC_NULL;
   int nback = (zcoord < nprocz-1) ? rank + nprocx * nprocy : MPI_PROC_NULL;
   //printf("%d:DEBUG -- nleft %d nrght %d nbot %d ntop %d nfrnt %d nback %d\n",rank,nleft,nrght,nbot,ntop,nfrnt,nback);

   int ibegin = imax *(xcoord  )/nprocx;
   int iend   = imax *(xcoord+1)/nprocx;
   int isize  = iend - ibegin;
   int jbegin = jmax *(ycoord  )/nprocy;
   int jend   = jmax *(ycoord+1)/nprocy;
   int jsize  = jend - jbegin;
   int kbegin = kmax *(zcoord  )/nprocz;
   int kend   = kmax *(zcoord+1)/nprocz;
   int ksize  = kend - kbegin;
   //printf("%d:DEBUG -- ibegin %d iend %d isize %d\n",rank,ibegin,iend,isize);
   //printf("%d:DEBUG -- jbegin %d jend %d jsize %d\n",rank,jbegin,jend,jsize);
   //printf("%d:DEBUG -- kbegin %d kend %d ksize %d\n",rank,kbegin,kend,ksize);

   struct neighs ngh;
   ngh.left = nleft;
   ngh.rght = nrght;
   ngh.bot  = nbot;
   ngh.top  = ntop;
   ngh.frnt = nfrnt;
   ngh.back = nback;

   struct sizes size;
   size.i = isize;
   size.j = jsize;
   size.k = ksize;

   struct procs nproc;
   nproc.x = nprocx;
   nproc.y = nprocy;
   nproc.z = nprocz;

   /* The halo update both updates the ghost cells and the boundary halo cells. To be precise with terminology,
    * the ghost cells only exist for multi-processor runs with MPI. The boundary halo cells are to set boundary
    * conditions. Halos refer to both the ghost cells and the boundary halo cells.
    */
   haloupdate_test(nhalo, corners, size, ngh, kmax, jmax, imax, nproc, do_timing);

   double*** xtmp;
   // This offsets the array addressing so that the real part of the array is from 0,0 to jsize,isize
   double*** x    = malloc3D(ksize+2*nhalo, jsize+2*nhalo, isize+2*nhalo, nhalo, nhalo, nhalo);
   double*** xnew = malloc3D(ksize+2*nhalo, jsize+2*nhalo, isize+2*nhalo, nhalo, nhalo, nhalo);

   if (! corners) { // need to initialize when not doing corners so there is no uninitialized memory
      for (int k = -nhalo; k < ksize+nhalo; k++){
         for (int j = -nhalo; j < jsize+nhalo; j++){
            for (int i = -nhalo; i < isize+nhalo; i++){
               x[k][j][i] = 0.0;
            }
         }
      }
   }

   for (int k = 0; k < ksize; k++){
      for (int j = 0; j < jsize; j++){
         for (int i = 0; i < isize; i++){
            x[k][j][i] = 5.0;
         }
      }
   }

   int ispan=5, jspan=5, kspan=5;
   if (ispan > imax/2) ispan = imax/2;
   if (jspan > jmax/2) jspan = jmax/2;
   if (kspan > kmax/2) kspan = kmax/2;
   for (int k = kmax/2 - kspan; k < kmax/2 + kspan; k++){
      for (int j = jmax/2 - jspan; j < jmax/2 + jspan; j++){
         for (int i = imax/2 - ispan; i < imax/2 + ispan; i++){
            if (k >= kbegin && k < kend && j >= jbegin && j < jend && i >= ibegin && i < iend) {
               x[k-kbegin][j-jbegin][i-ibegin] = 400.0;
            }
         }
      }
   }

   boundarycondition_update(x, nhalo, size, ngh);
   ghostcell_update(x, nhalo, corners, size, ngh, nproc, do_timing);

   for (int iter = 0; iter < 1000; iter++){
      cpu_timer_start(&tstart_stencil);

      for (int k = 0; k < ksize; k++){
         for (int j = 0; j < jsize; j++){
            for (int i = 0; i < isize; i++){
               xnew[k][j][i] = ( x[k][j][i] + x[k][j][i-1] + x[k][j][i+1] + x[k][j-1][i] + x[k][j+1][i] )/5.0;
            }
         }
      }

      SWAP_PTR(xnew, x, xtmp);

      stencil_time += cpu_timer_stop(tstart_stencil);

      boundarycondition_update(x, nhalo, size, ngh);
      ghostcell_update(x, nhalo, corners, size, ngh, nproc, do_timing);

      if (iter%100 == 0 && rank == 0) printf("Iter %d\n",iter);
   }
   total_time = cpu_timer_stop(tstart_total);

   Cartesian_print(x, kmax, jmax, imax, nhalo, nproc);

   if (rank == 0){
      printf("GhostExchange3D_ArrayAssign Timing is stencil %f boundary condition %f ghost cell %lf total %f\n",
             stencil_time,boundarycondition_time,ghostcell_time,total_time);
   }

   malloc3D_free(x, nhalo, nhalo, nhalo);
   malloc3D_free(xnew, nhalo, nhalo, nhalo);

   MPI_Finalize();
   exit(0);
}

void boundarycondition_update(double ***x, int nhalo, struct sizes size, struct neighs ngh)
{
   struct timespec tstart_boundarycondition;
   cpu_timer_start(&tstart_boundarycondition);

   int isize = size.i;
   int jsize = size.j;
   int ksize = size.k;

   int nleft = ngh.left;
   int nrght = ngh.rght;
   int nbot = ngh.bot;
   int ntop = ngh.top;
   int nfrnt = ngh.frnt;
   int nback = ngh.back;

// Boundary conditions -- constant
   if (nleft == MPI_PROC_NULL){
      for (int k = -nhalo; k < ksize+nhalo; k++){
         for (int j = -nhalo; j < jsize+nhalo; j++){
            for (int ll=-nhalo; ll<0; ll++){
               x[k][j][ll] = x[k][j][0];
            }
         }
      }
   }

   if (nrght == MPI_PROC_NULL){
      for (int k = -nhalo; k < ksize+nhalo; k++){
         for (int j = -nhalo; j < jsize+nhalo; j++){
            for (int ll=0; ll<nhalo; ll++){
               x[k][j][isize+ll] = x[k][j][isize-1];
            }
         }
      }
   }

   if (nbot == MPI_PROC_NULL){
      for (int k = -nhalo; k < ksize+nhalo; k++){
         for (int ll=-nhalo; ll<0; ll++){
            for (int i = -nhalo; i < isize+nhalo; i++){
               x[k][ll][i] = x[k][0][i];
            }
         }
      }
   }
      
   if (ntop == MPI_PROC_NULL){
      for (int k = -nhalo; k < ksize+nhalo; k++){
         for (int ll=0; ll<nhalo; ll++){
            for (int i = -nhalo; i < isize+nhalo; i++){
               x[k][jsize+ll][i] = x[k][jsize-1][i];
            }
         }
      }
   }

   if (nfrnt == MPI_PROC_NULL){
      for (int ll=-nhalo; ll<0; ll++){
         for (int j = -nhalo; j < jsize+nhalo; j++){
            for (int i = -nhalo; i < isize+nhalo; i++){
               x[ll][j][i] = x[0][j][i];
            }
         }
      }
   }
      
   if (nback == MPI_PROC_NULL){
      for (int ll=0; ll<nhalo; ll++){
         for (int j = -nhalo; j < jsize+nhalo; j++){
            for (int i = -nhalo; i < isize+nhalo; i++){
               x[ksize+ll][j][i] = x[ksize-1][j][i];
            }
         }
      }
   }

   boundarycondition_time += cpu_timer_stop(tstart_boundarycondition);
}

void ghostcell_update(double ***x, int nhalo, int corners, struct sizes size, struct neighs ngh, struct procs nproc, int do_timing)
{
   if (do_timing) MPI_Barrier(MPI_COMM_WORLD);

   int isize = size.i;
   int jsize = size.j;
   int ksize = size.k;

   int nprocx = nproc.x;
   int nprocy = nproc.y;
   int nprocz = nproc.z;

   int nleft = ngh.left;
   int nrght = ngh.rght;
   int nbot = ngh.bot;
   int ntop = ngh.top;
   int nfrnt = ngh.frnt;
   int nback = ngh.back;

   int icount, bufsize;

   struct timespec tstart_ghostcell;
   cpu_timer_start(&tstart_ghostcell);

   MPI_Request request[4];
   MPI_Status status[4];

   int jlow=0, jhgh=jsize;
   if (corners) {
      jlow = -nhalo;
      jhgh = jsize+nhalo;
   }
   int jnum = jhgh-jlow;
   int klow, khgh, knum;
   if (corners) {
      klow = -nhalo;
      khgh = ksize+nhalo;
      knum = ksize+2*nhalo;
   } else {
      klow = 0;
      khgh = ksize;
      knum = ksize;
   }
   bufsize = knum*jnum*nhalo;

   double xbuf_left_send[bufsize];
   double xbuf_rght_send[bufsize];
   double xbuf_rght_recv[bufsize];
   double xbuf_left_recv[bufsize];

   if (nleft != MPI_PROC_NULL){
      icount = 0;
      for (int k = klow; k < khgh; k++){
         for (int j = jlow; j < jhgh; j++){
            for (int ll = 0; ll < nhalo; ll++){
               xbuf_left_send[icount++] = x[k][j][ll];
            }
         }
      }
   }
   if (nrght != MPI_PROC_NULL){
      icount = 0;
      for (int k = klow; k < khgh; k++){
         for (int j = jlow; j < jhgh; j++){
            for (int ll = 0; ll < nhalo; ll++){
               xbuf_rght_send[icount++] = x[k][j][isize-nhalo+ll];
            }
         }
      }
   }

   MPI_Irecv(&xbuf_rght_recv, bufsize, MPI_DOUBLE, nrght, 1001, MPI_COMM_WORLD, &request[0]);
   MPI_Isend(&xbuf_left_send, bufsize, MPI_DOUBLE, nleft, 1001, MPI_COMM_WORLD, &request[1]);

   MPI_Irecv(&xbuf_left_recv, bufsize, MPI_DOUBLE, nleft, 1002, MPI_COMM_WORLD, &request[2]);
   MPI_Isend(&xbuf_rght_send, bufsize, MPI_DOUBLE, nrght, 1002, MPI_COMM_WORLD, &request[3]);
   MPI_Waitall(4, request, status);

   if (nrght != MPI_PROC_NULL){
      icount = 0;
      for (int k = klow; k < khgh; k++){
         for (int j = jlow; j < jhgh; j++){
            for (int ll = 0; ll < nhalo; ll++){
               x[k][j][isize+ll] = xbuf_rght_recv[icount++];
            }
         }
      }
   }
   if (nleft != MPI_PROC_NULL){
      icount = 0;
      for (int k = klow; k < khgh; k++){
         for (int j = jlow; j < jhgh; j++){
            for (int ll = 0; ll < nhalo; ll++){
               x[k][j][-nhalo+ll] = xbuf_left_recv[icount++];
            }
         }
      }
   }

   int ilow, ihgh, inum;
   if (corners) {
      ilow = -nhalo;
      ihgh = isize+nhalo;
      inum = isize+2*nhalo;
   } else {
      ilow = 0;
      ihgh = isize;
      inum = isize;
   }
   bufsize = knum*inum*nhalo;

   double xbuf_bot_send[bufsize];
   double xbuf_top_send[bufsize];
   double xbuf_top_recv[bufsize];
   double xbuf_bot_recv[bufsize];

   if (nbot != MPI_PROC_NULL){
      icount = 0;
      for (int k = klow; k < khgh; k++){
         for (int ll = 0; ll < nhalo; ll++){
            for (int i = ilow; i < ihgh; i++){
               xbuf_bot_send[icount++] = x[k][ll][i];
            }
         }
      }
   }
   if (ntop != MPI_PROC_NULL){
      icount = 0;
      for (int k = klow; k < khgh; k++){
         for (int ll = 0; ll < nhalo; ll++){
            for (int i = ilow; i < ihgh; i++){
               xbuf_top_send[icount++] = x[k][jsize-nhalo+ll][i];
            }
         }
      }
   }

   MPI_Irecv(&xbuf_top_recv, bufsize, MPI_DOUBLE, ntop, 1003, MPI_COMM_WORLD, &request[0]);
   MPI_Isend(&xbuf_bot_send, bufsize, MPI_DOUBLE, nbot, 1003, MPI_COMM_WORLD, &request[1]);

   MPI_Irecv(&xbuf_bot_recv, bufsize, MPI_DOUBLE, nbot, 1004, MPI_COMM_WORLD, &request[2]);
   MPI_Isend(&xbuf_top_send, bufsize, MPI_DOUBLE, ntop, 1004, MPI_COMM_WORLD, &request[3]);
   MPI_Waitall(4, request, status);

   if (ntop != MPI_PROC_NULL){
      icount = 0;
      for (int k = klow ; k < khgh; k++){
         for (int ll = 0; ll < nhalo; ll++){
            for (int i = ilow; i < ihgh; i++){
               x[k][jsize+ll][i] = xbuf_top_recv[icount++];
            }
         }
      }
   }
   if (nbot != MPI_PROC_NULL){
      icount = 0;
      for (int k = klow; k < khgh; k++){
         for (int ll = 0; ll < nhalo; ll++){
            for (int i = ilow; i < ihgh; i++){
               x[k][-nhalo+ll][i] = xbuf_bot_recv[icount++];
            }
         }
      }
   }

   if (corners) {
      jlow = -nhalo;
      jhgh = jsize+nhalo;
      jnum = jsize+2*nhalo;
   } else {
      jlow = 0;
      jhgh = jsize;
      jnum = jsize;
   }
   bufsize = jnum*inum*nhalo;

   double xbuf_frnt_send[bufsize];
   double xbuf_back_send[bufsize];
   double xbuf_frnt_recv[bufsize];
   double xbuf_back_recv[bufsize];

   if (nfrnt != MPI_PROC_NULL){
      icount = 0;
      for (int ll = 0; ll < nhalo; ll++){
         for (int j = jlow; j < jhgh; j++){
            for (int i = ilow; i < ihgh; i++){
               xbuf_frnt_send[icount++] = x[ll][j][i];
            }
         }
      }
   }
   if (ngh.back != MPI_PROC_NULL){
      icount = 0;
      for (int ll = 0; ll < nhalo; ll++){
         for (int j = jlow; j < jhgh; j++){
            for (int i = ilow; i < ihgh; i++){
               xbuf_back_send[icount++] = x[ksize-nhalo+ll][j][i];
            }
         }
      }
   }

   MPI_Irecv(&xbuf_back_recv, bufsize, MPI_DOUBLE, nback, 1005, MPI_COMM_WORLD, &request[0]);
   MPI_Isend(&xbuf_frnt_send, bufsize, MPI_DOUBLE, nfrnt, 1005, MPI_COMM_WORLD, &request[1]);

   MPI_Irecv(&xbuf_frnt_recv, bufsize, MPI_DOUBLE, nfrnt, 1006, MPI_COMM_WORLD, &request[2]);
   MPI_Isend(&xbuf_back_send, bufsize, MPI_DOUBLE, nback, 1006, MPI_COMM_WORLD, &request[3]);
   MPI_Waitall(4, request, status);

   if (nback != MPI_PROC_NULL){
      icount = 0;
      for (int ll = 0; ll < nhalo; ll++){
         for (int j = jlow; j < jhgh; j++){
            for (int i = ilow; i < ihgh; i++){
               x[ksize+ll][j][i] = xbuf_back_recv[icount++];
            }
         }
      }
   }
   if (nfrnt != MPI_PROC_NULL){
      icount = 0;
      for (int ll = 0; ll < nhalo; ll++){
         for (int j = jlow; j < jhgh; j++){
            for (int i = ilow; i < ihgh; i++){
               x[-nhalo+ll][j][i] = xbuf_frnt_recv[icount++];
            }
         }
      }
   }

   if (do_timing) MPI_Barrier(MPI_COMM_WORLD);

   ghostcell_time += cpu_timer_stop(tstart_ghostcell);
}

void haloupdate_test(int nhalo, int corners, struct sizes size, struct neighs ngh,
      int kmax, int jmax, int imax, struct procs nproc, int do_timing)
{
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   int isize = size.i;
   int jsize = size.j;
   int ksize = size.k;

   double*** x = malloc3D(ksize+2*nhalo, jsize+2*nhalo, isize+2*nhalo, nhalo, nhalo, nhalo);

   if (jsize > 12 || isize > 12) return;

   for (int k = -nhalo; k < ksize+nhalo; k++){
      for (int j = -nhalo; j < jsize+nhalo; j++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            x[k][j][i] = 0.0;
         }
      }
   }

   for (int k = 0; k < ksize; k++){
      for (int j = 0; j < jsize; j++){
         for (int i = 0; i < isize; i++){
            x[k][j][i] = rank * 1000 + k*100 + j*10 + i;
         }
      }
   }

   boundarycondition_update(x, nhalo, size, ngh);
   ghostcell_update(x, nhalo, corners, size, ngh, nproc, do_timing);

   Cartesian_print(x, kmax, jmax, imax, nhalo, nproc);

   malloc3D_free(x, nhalo, nhalo, nhalo);
}

void parse_input_args(int argc, char **argv, int &kmax, int &jmax, int &imax, 
                      int &nprocz, int &nprocy, int &nprocx, int &nhalo, int &corners, int &do_timing)
{
   int c;
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   while ((c = getopt(argc, argv, "ch:i:j:k:tx:y:z:")) != -1){
      switch(c){
         case 'c':
            corners = 1;
            break;
         case 'h':
            nhalo = atoi(optarg);
            break;
         case 'i':
            imax = atoi(optarg);
            break;
         case 'j':
            jmax = atoi(optarg);
            break;
         case 'k':
            kmax = atoi(optarg);
            break;
         case 't':
            do_timing = 1;
            break;
         case 'x':
            nprocx = atoi(optarg);
            break;
         case 'y':
            nprocy = atoi(optarg);
            break;
         case 'z':
            nprocz = atoi(optarg);
            break;
         case '?':
            if (optopt == 'h' || optopt == 'i' || optopt == 'j' || optopt == 'k' ||
                optopt == 'x' || optopt == 'y' || optopt == 'z'){
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
   int ycoord = (rank/nprocx)%nprocy;
   int zcoord = rank/(nprocx*nprocy);

   int ibegin = imax *(xcoord  )/nprocx;
   int iend   = imax *(xcoord+1)/nprocx;
   int isize  = iend - ibegin;
   int jbegin = jmax *(ycoord  )/nprocy;
   int jend   = jmax *(ycoord+1)/nprocy;
   int jsize  = jend - jbegin;
   int kbegin = kmax *(zcoord  )/nprocz;
   int kend   = kmax *(zcoord+1)/nprocz;
   int ksize  = kend - kbegin;

   int ierr = 0, ierr_global;
   if (isize < nhalo || jsize < nhalo || ksize < nhalo) {
      if (rank == 0) printf("Error -- local size of grid is less than the halo size\n");
      ierr = 1;
   }
   MPI_Allreduce(&ierr, &ierr_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
   if (ierr_global > 0) {
      MPI_Finalize();
      exit(0);
   }
}

void Cartesian_print(double ***x, int kmax, int jmax, int imax, int nhalo, struct procs nproc)
{
   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   int nprocx = nproc.x;
   int nprocy = nproc.y;
   int nprocz = nproc.z;

   int isize_total=0;
   int isizes[nprocx];
   for (int ii = 0; ii < nprocx; ii++){
      int xcoord = rank%nprocx;
      int ycoord = rank/nprocx%nprocy;
      int zcoord = rank/(nprocx*nprocy);
      int ibegin = imax *(xcoord  )/nprocx;
      int iend   = imax *(xcoord+1)/nprocx;
      isizes[ii] = iend-ibegin;
      isize_total += isizes[ii] + 2*nhalo;
   }

   if (isize_total > 40) return;

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
   int ycoord = rank/nprocx%nprocy;
   int zcoord = rank/(nprocx*nprocy);

   int ibegin = imax *(xcoord  )/nprocx;
   int iend   = imax *(xcoord+1)/nprocx;
   int isize  = iend - ibegin;
   int jbegin = jmax *(ycoord  )/nprocy;
   int jend   = jmax *(ycoord+1)/nprocy;
   int jsize  = jend - jbegin;
   int kbegin = kmax *(zcoord  )/nprocz;
   int kend   = kmax *(zcoord+1)/nprocz;
   int ksize  = kend - kbegin;

   double *xrow = (double *)malloc(isize_total*sizeof(double));
   for (int kk=0; kk < nprocz; kk++){
      int klen = 0;
      int klen_max;
      if (zcoord == kk) {
         klen = ksize;
      }
      MPI_Allreduce(&klen,&klen_max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      for (int k=-nhalo; k<klen_max+nhalo; k++){
         for (int jj=nprocy-1; jj >= 0; jj--){
            int ilen = 0;
            int jlen = 0;
            int jlen_max;
            int *ilen_global = (int *)malloc(nprocs*sizeof(int));
            int *ilen_displ = (int *)malloc(nprocs*sizeof(int));
            if (zcoord == kk && ycoord == jj) {
               ilen = isize + 2*nhalo;
               jlen = jsize;
            }
            MPI_Allgather(&ilen,1,MPI_INT,ilen_global,1,MPI_INT,MPI_COMM_WORLD);
            MPI_Allreduce(&jlen,&jlen_max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
            ilen_displ[0] = 0;
            for (int i=1; i<nprocs; i++){
               ilen_displ[i] = ilen_displ[i-1] + ilen_global[i-1];
            }
            if (rank == 0) printf(" nprocz %d k %d nprocy %d\n",kk,k,jj);
            for (int j=jlen_max+nhalo-1; j>=-nhalo; j--){
               MPI_Gatherv(&x[k][j][-nhalo],ilen,MPI_DOUBLE,xrow,ilen_global,ilen_displ,MPI_DOUBLE,0,MPI_COMM_WORLD);
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
         if (rank == 0) printf("\n");
      }
   }
   if (rank == 0) printf("k-i plane\n");
   for (int jj=0; jj < nprocy; jj++){
      int jlen = 0;
      int jlen_max;
      if (ycoord == jj) {
         jlen = jsize;
      }
      MPI_Allreduce(&jlen,&jlen_max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
      for (int j=-nhalo; j<jlen_max+nhalo; j++){
         for (int kk=nprocz-1; kk >= 0; kk--){
            int ilen = 0;
            int klen = 0;
            int klen_max;
            int *ilen_global = (int *)malloc(nprocs*sizeof(int));
            int *ilen_displ = (int *)malloc(nprocs*sizeof(int));
            if (zcoord == kk && ycoord == jj) {
               ilen = isize + 2*nhalo;
               klen = ksize;
            }
            MPI_Allgather(&ilen,1,MPI_INT,ilen_global,1,MPI_INT,MPI_COMM_WORLD);
            MPI_Allreduce(&klen,&klen_max,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
            ilen_displ[0] = 0;
            for (int i=1; i<nprocs; i++){
               ilen_displ[i] = ilen_displ[i-1] + ilen_global[i-1];
            }
            if (rank == 0) printf(" nprocy %d j %d nprocz %d\n",jj,j,kk);
            for (int k=klen_max+nhalo-1; k>=-nhalo; k--){
               MPI_Gatherv(&x[k][j][-nhalo],ilen,MPI_DOUBLE,xrow,ilen_global,ilen_displ,MPI_DOUBLE,0,MPI_COMM_WORLD);
               if (rank == 0) {
                  printf("%3d:",k);
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
      }
   }
   free(xrow);
}
