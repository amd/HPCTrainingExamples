#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <hip/hip_runtime.h>
#include <roctx.h>

#include "timer.h"

#define SWAP_PTR(xnew,xold,xtmp) (xtmp=xnew, xnew=xold, xold=xtmp)

#define xv(j,i) x[(j+nhalo)*jstride+(i+nhalo)]
#define xvnew(j,i) xnew[(j+nhalo)*jstride+(i+nhalo)]

#define HIP_CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
      }\
}

__global__ void init_core (double *x, int nhalo, int jsize, int isize, double val, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x -nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y -nhalo;
    if (tidy < jsize+nhalo && tidx < isize+nhalo) {
        xv(tidy,tidx) = val;
    }
}

__global__ void init_core2 (double *x, int jmax, int jspan, int jbegin, int jend, int imax, int ispan, int ibegin, int iend, int rank, int jstride, int nhalo)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x + imax/2 - ispan;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jmax/2 - jspan;
    if (tidy < jmax/2 + jspan && tidx < imax/2 + ispan) {
        if (tidy >= jbegin && tidy < jend && tidx >= ibegin && tidx < iend) {
           xv(tidy-jbegin,tidx-ibegin) = 400.0 + 0.1 * (double)(rank + tidy);
        }
    }
}

__global__ void blur (double *x, double *xnew, int jsize, int isize, int jstride, int nhalo)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize && tidx < isize) {
        xvnew(tidy,tidx) = ( xv(tidy,tidx) + xv(tidy,tidx-1) + xv(tidy,tidx+1) + xv(tidy-1,tidx) + xv(tidy+1,tidx) )/5.0;
    }
}

__global__ void blur_inner (double *x, double *xnew, int jsize, int isize, int jstride, int nhalo)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x + nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + nhalo;
    if (tidy < jsize - nhalo && tidx < isize - nhalo) {
        xvnew(tidy,tidx) = ( xv(tidy,tidx) + xv(tidy,tidx-1) + xv(tidy,tidx+1) + xv(tidy-1,tidx) + xv(tidy+1,tidx) )/5.0;
    }
}

__global__ void blur_outer_left (double *x, double *xnew, int jsize, int isize, int jstride, int nhalo)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize && tidx < nhalo) {
        xvnew(tidy,tidx) = ( xv(tidy,tidx) + xv(tidy,tidx-1) + xv(tidy,tidx+1) + xv(tidy-1,tidx) + xv(tidy+1,tidx) )/5.0;
    }
}

__global__ void blur_outer_right (double *x, double *xnew, int jsize, int isize, int jstride, int nhalo)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x + isize - nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize && tidx < isize) {
        xvnew(tidy,tidx) = ( xv(tidy,tidx) + xv(tidy,tidx-1) + xv(tidy,tidx+1) + xv(tidy-1,tidx) + xv(tidy+1,tidx) )/5.0;
    }
}

__global__ void blur_outer_top (double *x, double *xnew, int jsize, int isize, int jstride, int nhalo)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x + nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jsize - nhalo;
    if (tidy < jsize && tidx < isize - nhalo) {
        xvnew(tidy,tidx) = ( xv(tidy,tidx) + xv(tidy,tidx-1) + xv(tidy,tidx+1) + xv(tidy-1,tidx) + xv(tidy+1,tidx) )/5.0;
    }
}

__global__ void blur_outer_bottom (double *x, double *xnew, int jsize, int isize, int jstride, int nhalo)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x + nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < nhalo && tidx < isize- nhalo) {
        xvnew(tidy,tidx) = ( xv(tidy,tidx) + xv(tidy,tidx-1) + xv(tidy,tidx+1) + xv(tidy-1,tidx) + xv(tidy+1,tidx) )/5.0;
    }
}

__global__ void enforce_bcs_left (double *x, int nhalo, int jsize, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x - nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize && tidx < 0) {
        xv(tidy,tidx) = xv(tidy,0);
    }
}

__global__ void enforce_bcs_rght (double *x, int nhalo, int jsize, int isize, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < jsize  && tidx < nhalo) {
        xv(tidy,isize+tidx) = xv(tidy,isize-1);
    }
}

__global__ void enforce_bcs_bot (double *x, int nhalo, int isize, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x - nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y - nhalo;
    if (tidy < 0 && tidx < isize + nhalo) {
        xv(tidy,tidx) = xv(0,tidx);
    }
}

__global__ void enforce_bcs_top (double *x, int nhalo, int jsize, int isize, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x - nhalo;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y;
    if (tidy < nhalo && tidx < isize + nhalo) {
        xv(jsize+tidy,tidx) = xv(jsize-1,tidx);
    }
}

__global__ void ghost_left_send (double *xbuf_left_send, double *x, int jlow, int jhgh, int nhalo, int bufcount, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jlow;
    int offset = (tidy - jlow) * nhalo + tidx;
    if (tidy < jhgh && tidx < nhalo && offset < bufcount) {
        xbuf_left_send[offset] = xv(tidy,tidx);
    }
}

__global__ void ghost_rght_send (double *xbuf_rght_send, double *x, int jlow, int jhgh, int isize, int nhalo, int bufcount, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jlow;
    int offset = (tidy - jlow) * nhalo + tidx;
    if (tidy < jhgh && tidx < nhalo && offset < bufcount) {
        xbuf_rght_send[offset] = xv(tidy,isize - nhalo + tidx);
    }
}

__global__ void ghost_rght_recv (double *xbuf_rght_recv, double *x, int jlow, int jhgh, int isize, int nhalo, int bufcount, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jlow;
    int offset = (tidy - jlow) * nhalo + tidx;
    if (tidy < jhgh && tidx < nhalo && offset < bufcount) {
        xv(tidy,isize + tidx) = xbuf_rght_recv[offset];
    }
}

__global__ void ghost_left_recv (double *xbuf_left_recv, double *x, int jlow, int jhgh, int nhalo, int bufcount, int jstride)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jlow;
    int offset = (tidy - jlow) * nhalo + tidx;
    if (tidy < jhgh && tidx < nhalo && offset < bufcount) {
        xv(tidy,tidx - nhalo) = xbuf_left_recv[offset];
    }
}

void parse_input_args(int argc, char **argv, int &jmax, int &imax, int &nprocy, int &nprocx, int &nhalo, int &corners, int &maxIter, int &do_timing, int &do_print);
void Cartesian_print(double *x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int jstride);
void boundarycondition_update(double *x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop);
void ghostcell_update(double *x, int nhalo, int corners, int jsize, int isize,
      int nleft, int nrght, int nbot, int ntop, int do_timing);
void haloupdate_test(int nhalo, int corners, int jsize, int isize, int nleft, int nrght, int nbot, int ntop,
      int jmax, int imax, int nprocy, int nprocx, int do_timing, int do_print);

double boundarycondition_time=0.0, ghostcell_time=0.0;

double *xbuf_left_send, *xbuf_rght_send, *xbuf_rght_recv, *xbuf_left_recv;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (rank == 0) printf("Parallel run with no threads\n");
   hipSetDevice(0);

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

   int jstride = isize+2*nhalo;

   int jlow=0, jhgh=jsize;
   if (corners) {
      if (nbot == MPI_PROC_NULL) jlow = -nhalo;
      if (ntop == MPI_PROC_NULL) jhgh = jsize+nhalo;
   }
   int jnum = jhgh-jlow;
   int bufcount = jnum*nhalo;

   roctxRangePush("BufAlloc");
   HIP_CHECK(hipMalloc(&xbuf_left_send, bufcount*sizeof(double)));
   HIP_CHECK(hipMalloc(&xbuf_rght_send, bufcount*sizeof(double)));
   HIP_CHECK(hipMalloc(&xbuf_rght_recv, bufcount*sizeof(double)));
   HIP_CHECK(hipMalloc(&xbuf_left_recv, bufcount*sizeof(double)));
   roctxRangePop(); //BufAlloc

   /* The halo update both updates the ghost cells and the boundary halo cells. To be precise with terminology,
    * the ghost cells only exist for multi-processor runs with MPI. The boundary halo cells are to set boundary
    * conditions. Halos refer to both the ghost cells and the boundary halo cells.
    */
   haloupdate_test(nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, jmax, imax, nprocy, nprocx, do_timing, do_print);

   double* xtmp;
   // Using 1D representation of a 2D array with nhalo offsets
   int totcells=(jsize+2*nhalo)*(isize+2*nhalo);
   double *x=NULL, *xnew=NULL;
   HIP_CHECK(hipMalloc(&x, totcells * sizeof(double)));
   HIP_CHECK(hipMalloc(&xnew, totcells * sizeof(double)));

   double *x_h=NULL;
   if (do_print) {
      // Allocate a host array once for use in printing/debugging
      x_h = (double *) malloc (totcells * sizeof(double));
   }

   if (! corners) { // need to initialize when not doing corners so there is no uninitialized memory
      dim3 grid((isize+2*nhalo+63)/64, (jsize+2*nhalo+3)/4, 1);
      dim3 block(64, 4, 1);
      init_core <<< grid, block >>> (x, nhalo, jsize, isize, 0.0, jstride);
      hipDeviceSynchronize();
   }

   dim3 grid((isize+63)/64, (jsize+3)/4, 1);
   dim3 block(64, 4, 1);
   double val = static_cast<double>(rank) + 5.0;
   init_core <<< grid, block >>> (x, nhalo, jsize, isize, val, jstride);
   hipDeviceSynchronize();

   int ispan=5, jspan=5;
   if (ispan > imax/2) ispan = imax/2;
   if (jspan > jmax/2) jspan = jmax/2;
   dim3 grid2((2*ispan+63)/64, (2*jspan+3)/4, 1);
   dim3 block2(64, 4, 1);
   init_core2 <<< grid2, block2 >>> (x, jmax, jspan, jbegin, jend, imax, ispan, ibegin, iend, rank, jstride, nhalo);
   hipDeviceSynchronize();

   boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);

   if (do_print == 1) {
      if (rank == 0) printf("Initial State \n");
      // note: this ghostcell_update is only for printing purposes to show that the
      // solution matches the one produced by Orig also on the halo for debugging
      // the value of the solution on the halo does not matter as long as
      // it is up to date when it is used to compute the solution on cells that need it
      ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);
      HIP_CHECK(hipMemcpy(x_h, x, totcells * sizeof(double), hipMemcpyDefault));
      Cartesian_print(x_h, jmax, imax, nhalo, nprocy, nprocx, jstride);
   }

   roctxMark("Starting main iteration loop");

   for (int iter = 0; iter < maxIter; iter++){
      cpu_timer_start(&tstart_stencil);
      roctxRangePush("InnerCellsUpdate");

      // note: we assume that the halo size is greater than or equal
      // to the the operator's stencil reach (which for the blur is 1)
      blur_inner <<< grid, block >>> (x, xnew, jsize, isize, jstride, nhalo);
      roctxRangePop(); //InnerCellsUpdate
      stencil_time += cpu_timer_stop(tstart_stencil);
      roctxRangePush("GhostCellUpdate");
      ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);
      roctxRangePop(); //GhostCellUpdate
      MPI_Barrier(MPI_COMM_WORLD);
      hipDeviceSynchronize();

      cpu_timer_start(&tstart_stencil);
      roctxRangePush("OuterCellsUpdate");
      blur_outer_left <<< grid, block >>> (x, xnew, jsize, isize, jstride, nhalo);
      blur_outer_right <<< grid, block >>> (x, xnew, jsize, isize, jstride, nhalo);
      blur_outer_top <<< grid, block >>> (x, xnew, jsize, isize, jstride, nhalo);
      blur_outer_bottom <<< grid, block >>> (x, xnew, jsize, isize, jstride, nhalo);
      roctxRangePop(); //OuterCellsUpdate
      stencil_time += cpu_timer_stop(tstart_stencil);

      roctxRangePush("BoundaryUpdate");
      boundarycondition_update(xnew, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
      roctxRangePop(); //BoundaryUpdate

      if (do_print == 1) {
         // halo update on soln
         // NOTE: this is just for visualziation
         // we do not care in general what the solution is in the halo
         // as long as it is correct when we need it to compute the next
         // value of the solution
         ghostcell_update(xnew, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);
      }

      SWAP_PTR(xnew, x, xtmp);

      if (iter%10 == 0 && rank == 0) printf("Iter %d\n",iter);
      if (do_print == 1) {
         HIP_CHECK(hipMemcpy(x_h, x, totcells * sizeof(double), hipMemcpyDefault));
         Cartesian_print(x_h, jmax, imax, nhalo, nprocy, nprocx, jstride);
      }
   }

   roctxMark("Stopping main iteration loop");

   total_time = cpu_timer_stop(tstart_total);

   if (rank == 0){
      printf("GhostExchange_ArrayAssign_HIP Timing is stencil %f boundary condition %f ghost cell %lf total %f\n",
             stencil_time,boundarycondition_time,ghostcell_time,total_time);
   }

   HIP_CHECK(hipFree(x));
   HIP_CHECK(hipFree(xnew));

   if (do_print) {
      free(x_h);
   }

   HIP_CHECK(hipFree(xbuf_left_send));
   HIP_CHECK(hipFree(xbuf_rght_send));
   HIP_CHECK(hipFree(xbuf_rght_recv));
   HIP_CHECK(hipFree(xbuf_left_recv));

   MPI_Finalize();
   exit(0);
}

void boundarycondition_update(double **x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop)
{
   struct timespec tstart_boundarycondition;
   cpu_timer_start(&tstart_boundarycondition);

   dim3 gridVrt((nhalo+3)/4, (jsize+63)/64, 1);
   dim3 blockVrt(4, 64, 1);
   dim3 gridHor((isize+2*nhalo+63)/64, (nhalo+3)/4, 1);
   dim3 blockHor(64, 4, 1);

// Boundary conditions -- constant
   if (nleft == MPI_PROC_NULL){

      enforce_bcs_left <<< gridVrt, blockVrt >>> (x, nhalo, jsize);
      hipDeviceSynchronize();

   }

   if (nrght == MPI_PROC_NULL){

      enforce_bcs_rght <<< gridVrt, blockVrt >>> (x, nhalo, jsize, isize);
      hipDeviceSynchronize();

   }

   if (nbot == MPI_PROC_NULL){

      enforce_bcs_bot <<< gridHor, blockHor >>> (x, nhalo, isize);
      hipDeviceSynchronize();

   }
      
   if (ntop == MPI_PROC_NULL){

      enforce_bcs_top <<< gridHor, blockHor >>> (x, nhalo, jsize, isize);
      hipDeviceSynchronize();

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

   // NOTE: this function has not been ported to HIP
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

   if (do_print) {
      double *x_h = (double *) malloc (totcells*sizeof(double));
      HIP_CHECK(hipMemcpy(x_h, x, totcells*sizeof(double), hipMemcpyDefault));
      Cartesian_print(x_h, jmax, imax, nhalo, nprocy, nprocx, jstride);
      free(x_h);
   }

   HIP_CHECK(hipFree(x));

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
