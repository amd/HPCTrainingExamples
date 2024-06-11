#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <roctx.h>

#include "timer.h"

//#pragma omp requires unified_shared_memory

#define SWAP_PTR(xnew,xold,xtmp) (xtmp=xnew, xnew=xold, xold=xtmp)

#define xv(j,i) x[(j+nhalo)*jstride+(i+nhalo)]
#define xvnew(j,i) xnew[(j+nhalo)*jstride+(i+nhalo)]

void parse_input_args(int argc, char **argv, int &jmax, int &imax, int &nprocy, int &nprocx, int &nhalo, int &corners, int &maxIter, int &do_timing);
void Cartesian_print(double *x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int jstride, int totcells);
void boundarycondition_update(double *x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop);
void ghostcell_update(double *x, int nhalo, int corners, int jsize, int isize,
      int nleft, int nrght, int nbot, int ntop, int do_timing);
void haloupdate_test(int nhalo, int corners, int jsize, int isize, int nleft, int nrght, int nbot, int ntop,
      int jmax, int imax, int nprocy, int nprocx, int do_timing);

double boundarycondition_time=0.0, ghostcell_time=0.0;

double *xbuf_left_send, *xbuf_rght_send, *xbuf_rght_recv, *xbuf_left_recv;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (rank == 0) printf("Parallel run with no threads\n");

//   int numPerNUMA=nprocs/8;
//   int DevNum;
//   switch(rank/numPerNUMA)
//   {
//     case 0:
//      DevNum = 3;
//      break;
//     case 1:
//      DevNum = 2;
//      break;
//     case 2:
//      DevNum = 1;
//      break;
//     case 3:
//      DevNum = 0;
//      break;
//     case 4:
//      DevNum = 7;
//      break;
//     case 5:
//      DevNum = 6;
//      break;
//     case 6:
//      DevNum = 5;
//      break;
//     case 7:
//      DevNum = 4;
//      break;
//   }
//   omp_set_default_device(DevNum);
//
//   if (rank == 0 || rank == 1){
//      omp_set_default_device(4);
//   }

   int imax = 2000, jmax = 2000;
   int nprocx = 0, nprocy = 0;
   int nhalo = 2, corners = 0;
   int do_timing = 0;
   int maxIter = 1000;

   parse_input_args(argc, argv, jmax, imax, nprocy, nprocx, nhalo, corners, maxIter, do_timing);
 
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

   int jstride = isize+2*nhalo;

   int jlow=0, jhgh=jsize;
   if (corners) {
      if (nbot == MPI_PROC_NULL) jlow = -nhalo;
      if (ntop == MPI_PROC_NULL) jhgh = jsize+nhalo;
   }
   int jnum = jhgh-jlow;
   int bufcount = jnum*nhalo;

   #pragma omp target
   {}

   roctxRangePush("BufAlloc");
   xbuf_left_send = (double *)malloc(bufcount*sizeof(double));
   xbuf_rght_send = (double *)malloc(bufcount*sizeof(double));
   xbuf_rght_recv = (double *)malloc(bufcount*sizeof(double));
   xbuf_left_recv = (double *)malloc(bufcount*sizeof(double));
   #pragma omp target enter data map(alloc: xbuf_left_send[0:bufcount], xbuf_rght_send[0:bufcount]) 
   #pragma omp target enter data map(alloc: xbuf_rght_recv[0:bufcount], xbuf_left_recv[0:bufcount])
   roctxRangePop(); //BufAlloc

   /* The halo update both updates the ghost cells and the boundary halo cells. To be precise with terminology,
    * the ghost cells only exist for multi-processor runs with MPI. The boundary halo cells are to set boundary
    * conditions. Halos refer to both the ghost cells and the boundary halo cells.
    */
   haloupdate_test(nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, jmax, imax, nprocy, nprocx, do_timing);

   double* xtmp;
   // Using 1D representation of a 2D array with nhalo offsets
   int totcells=(jsize+2*nhalo)*(isize+2*nhalo);
   double *x    = (double *)malloc(totcells*sizeof(double));
   double *xnew = (double *)malloc(totcells*sizeof(double));
   #pragma omp target enter data map(alloc: x[0:totcells], xnew[0:totcells])

   if (! corners) { // need to initialize when not doing corners so there is no uninitialized memory
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = -nhalo; j < jsize+nhalo; j++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            xv(j,i) = 0.0;
         }
      }
   }

   #pragma omp target teams distribute parallel for collapse(2)
   for (int j = 0; j < jsize; j++){
      for (int i = 0; i < isize; i++){
         xv(j,i) = 5.0;
      }
   }

   int ispan=5, jspan=5;
   if (ispan > imax/2) ispan = imax/2;
   if (jspan > jmax/2) jspan = jmax/2;
   #pragma omp target teams distribute parallel for collapse(2)
   for (int j = jmax/2 - jspan; j < jmax/2 + jspan; j++){
      for (int i = imax/2 - ispan; i < imax/2 + ispan; i++){
         if (j >= jbegin && j < jend && i >= ibegin && i < iend) {
            xv(j-jbegin,i-ibegin) = 400.0;
         }
      }
   }

   boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
   ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);

   roctxMark("Starting main iteration loop");

   for (int iter = 0; iter < maxIter; iter++){
      roctxRangePush("Stencil");
      cpu_timer_start(&tstart_stencil);

      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = 0; j < jsize; j++){
         for (int i = 0; i < isize; i++){
            xvnew(j,i) = ( xv(j,i) + xv(j,i-1) + xv(j,i+1) + xv(j-1,i) + xv(j+1,i) )/5.0;
         }
      }
      roctxRangePop(); //Stencil

      SWAP_PTR(xnew, x, xtmp);

      stencil_time += cpu_timer_stop(tstart_stencil);

      roctxRangePush("BoundaryUpdate");
      boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
      roctxRangePop(); //BoundaryUpdate
      roctxRangePush("GhostCellUpdate");
      ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);
      roctxRangePop(); //GhostCellUpdate

      if (iter%100 == 0 && rank == 0) printf("Iter %d\n",iter);
   }

   roctxMark("Stopping main iteration loop");

   total_time = cpu_timer_stop(tstart_total);

   Cartesian_print(x, jmax, imax, nhalo, nprocy, nprocx, jstride, totcells);

   if (rank == 0){
      printf("GhostExchange_ArrayAssign Timing is stencil %f boundary condition %f ghost cell %lf total %f\n",
             stencil_time,boundarycondition_time,ghostcell_time,total_time);
   }

   #pragma omp target exit data map(release: x, xnew)
   free(x);
   free(xnew);

   #pragma omp target exit data map(release: xbuf_left_send, xbuf_rght_send) 
   #pragma omp target exit data map(release: xbuf_rght_recv, xbuf_left_recv)
   free(xbuf_left_send);
   free(xbuf_rght_send);
   free(xbuf_rght_recv);
   free(xbuf_left_recv);

   MPI_Finalize();
   exit(0);
}

void boundarycondition_update(double *x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop)
{
   int jstride = isize+2*nhalo;

   struct timespec tstart_boundarycondition;
   cpu_timer_start(&tstart_boundarycondition);

// Boundary conditions -- constant
   if (nleft == MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = 0; j < jsize; j++){
         for (int ll=-nhalo; ll<0; ll++){
            xv(j,ll) = xv(j,0);
         }
      }
   }

   if (nrght == MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = 0; j < jsize; j++){
         for (int ll=0; ll<nhalo; ll++){
            xv(j,isize+ll) = xv(j,isize-1);
         }
      }
   }

   if (nbot == MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int ll=-nhalo; ll<0; ll++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            xv(ll,i) = xv(0,i);
         }
      }
   }
      
   if (ntop == MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int ll=0; ll<nhalo; ll++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            xv(jsize+ll,i) = xv(jsize-1,i);
         }
      }
   }

   boundarycondition_time += cpu_timer_stop(tstart_boundarycondition);
}

void ghostcell_update(double *x, int nhalo, int corners, int jsize, int isize, int nleft, int nrght, int nbot, int ntop, int do_timing)
{
   int jstride = isize+2*nhalo;

   //if (do_timing) MPI_Barrier(MPI_COMM_WORLD);

   struct timespec tstart_ghostcell;
   cpu_timer_start(&tstart_ghostcell);

   roctxRangePush("MPIRequest");
   MPI_Request request[4*nhalo];
   MPI_Status status[4*nhalo];
   roctxRangePop(); //MPIRequest

   int jlow=0, jhgh=jsize;
   if (corners) {
      if (nbot == MPI_PROC_NULL) jlow = -nhalo;
      if (ntop == MPI_PROC_NULL) jhgh = jsize+nhalo;
   }
   int jnum = jhgh-jlow;
   int bufcount = jnum*nhalo;
   int offset = 0;

   roctxRangePush("LoadLeftRight");
   if (nleft != MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
	    offset = (j - jlow) * nhalo + ll;
            xbuf_left_send[offset] = xv(j,ll);
         }
      }
   }
   if (nrght != MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xbuf_rght_send[offset] = xv(j,isize-nhalo+ll);
         }
      }
   }
   roctxRangePop(); //LoadLeftRight

   roctxRangePush("MPILeftRightExchange");
   MPI_Irecv(xbuf_rght_recv, bufcount, MPI_DOUBLE, nrght, 1001,
             MPI_COMM_WORLD, &request[0]);
   MPI_Isend(xbuf_left_send, bufcount, MPI_DOUBLE, nleft, 1001,
             MPI_COMM_WORLD, &request[1]);

   MPI_Irecv(xbuf_left_recv, bufcount, MPI_DOUBLE, nleft, 1002,
             MPI_COMM_WORLD, &request[2]);
   MPI_Isend(xbuf_rght_send, bufcount, MPI_DOUBLE, nrght, 1002,
             MPI_COMM_WORLD, &request[3]);
   MPI_Waitall(4, request, status);
   roctxRangePop(); //MPILeftRightExchange

   roctxRangePush("UnpackLeftRight");
   if (nrght != MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xv(j,isize+ll) = xbuf_rght_recv[offset];
         }
      }
   }
   if (nleft != MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xv(j,-nhalo+ll) = xbuf_left_recv[offset];
         }
      }
   }
   roctxRangePop(); //UnpackLeftRight

   roctxRangePush("MPIUpDownExchange");
   if (corners) {
      bufcount = nhalo*(isize+2*nhalo);
      MPI_Irecv(&xv(jsize,-nhalo),   bufcount, MPI_DOUBLE, ntop, 1001, MPI_COMM_WORLD, &request[0]);
      MPI_Isend(&xv(0    ,-nhalo),   bufcount, MPI_DOUBLE, nbot, 1001, MPI_COMM_WORLD, &request[1]);

      MPI_Irecv(&xv(     -nhalo,-nhalo), bufcount, MPI_DOUBLE, nbot, 1002, MPI_COMM_WORLD, &request[2]);
      MPI_Isend(&xv(jsize-nhalo,-nhalo), bufcount, MPI_DOUBLE, ntop, 1002, MPI_COMM_WORLD, &request[3]);
      MPI_Waitall(4, request, status);
   } else {
      for (int j = 0; j<nhalo; j++){
         MPI_Irecv(&xv(jsize+j,0),   isize, MPI_DOUBLE, ntop, 1001+j*2, MPI_COMM_WORLD, &request[0+j*4]);
         MPI_Isend(&xv(0+j    ,0),   isize, MPI_DOUBLE, nbot, 1001+j*2, MPI_COMM_WORLD, &request[1+j*4]);

         MPI_Irecv(&xv(     -nhalo+j,0), isize, MPI_DOUBLE, nbot, 1002+j*2, MPI_COMM_WORLD, &request[2+j*4]);
         MPI_Isend(&xv(jsize-nhalo+j,0), isize, MPI_DOUBLE, ntop, 1002+j*2, MPI_COMM_WORLD, &request[3+j*4]);
         }
      MPI_Waitall(4*nhalo, request, status);
   }
   roctxRangePop(); //MPIUpDownExchange

   //if (do_timing) MPI_Barrier(MPI_COMM_WORLD);

   ghostcell_time += cpu_timer_stop(tstart_ghostcell);
}

void haloupdate_test(int nhalo, int corners, int jsize, int isize, int nleft, int nrght, int nbot, int ntop,
      int jmax, int imax, int nprocy, int nprocx, int do_timing)
{
   int jstride = isize+2*nhalo;

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int totcells=(jsize+2*nhalo)*(isize+2*nhalo);
   double* x = (double *)malloc(totcells*sizeof(double));
   #pragma omp target enter data map(alloc: x[0:totcells])

   if (jsize > 12 || isize > 12) return;

   #pragma omp target teams distribute parallel for collapse(2)
   for (int j = -nhalo; j < jsize+nhalo; j++){
      for (int i = -nhalo; i < isize+nhalo; i++){
         xv(j,i) = 0.0;
      }
   }

   #pragma omp target teams distribute parallel for collapse(2)
   for (int j = 0; j < jsize; j++){
      for (int i = 0; i < isize; i++){
         xv(j,i) = rank * 1000 + j*10 + i;
      }
   }

   boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
   ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);

   Cartesian_print(x, jmax, imax, nhalo, nprocy, nprocx, jstride, totcells);

   #pragma omp target exit data map(release: x)
   free(x);
}

void parse_input_args(int argc, char **argv, int &jmax, int &imax, int &nprocy, int &nprocx, int &nhalo, int &corners, int &maxIter, int &do_timing)
{
   int c;
   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   while ((c = getopt(argc, argv, "ch:I:i:j:tx:y:")) != -1){
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

void Cartesian_print(double *x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int jstride, int totcells)
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

   if (isize_total > 40) return;
   #pragma omp target update from(x[0:totcells])

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
         MPI_Gatherv(&xv(j,-nhalo),ilen,MPI_DOUBLE,xrow,ilen_global,ilen_displ,MPI_DOUBLE,0,MPI_COMM_WORLD);
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
