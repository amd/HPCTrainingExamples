#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#ifdef USE_PNETCDF
#include <pnetcdf.h>
#endif
#include <cmath>
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

__global__ void init_core2 (double *x, int jmax, int jspan, int jbegin, int jend, int imax, int ispan, int ibegin, int iend, double sigma, int jstride, int nhalo)
{
    int tidx = threadIdx.x + blockIdx.x * blockDim.x + imax/2 - ispan;
    int tidy = threadIdx.y + blockIdx.y * blockDim.y + jmax/2 - jspan;
    if (tidy < jmax/2 + jspan && tidx < imax/2 + ispan) {
        if (tidy >= jbegin && tidy < jend && tidx >= ibegin && tidx < iend) {
           double x_phys = tidx + ibegin;
           double y_phys = tidy + jbegin;
           double x_center = imax / 2.0;
           double y_center = jmax / 2.0;
           xv([tidy-jbegin,tidx-ibegin) = exp(-0.5 * ( ((x_phys - x_center)*(x_phys - x_center)/(sigma*sigma)
                               + (y_phys - y_center)*(y_phys - y_center)/(sigma*sigma)) ));
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

void parse_input_args(int argc, char **argv, int &jmax, int &imax, int &nprocy, int &nprocx, int &nhalo, int &corners, int &maxIter, int &do_timing, int &do_print);
void Cartesian_print(double *x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int jstride);
#ifdef USE_PNETCDF
void create_netcdf_file(const char *fname, int jmax, int imax, MPI_Comm comm, int *ncid, int *varid, int *varid_xcoord, int *varid_ycoord);
void write_netcdf_soln(double *x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int tstep, int ncid, int varid);
void write_netcdf_coords(int imax, int jmax, int nprocx, int nprocy, double Lx, double Ly, int ncid, int varid_xcoord, int varid_ycoord);
#endif
void boundarycondition_update(double *x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop);
void ghostcell_update(double *x, int nhalo, int corners, int jsize, int isize,
      int nleft, int nrght, int nbot, int ntop, int do_timing);
void haloupdate_test(int nhalo, int corners, int jsize, int isize, int nleft, int nrght, int nbot, int ntop,
      int jmax, int imax, int nprocy, int nprocx, int do_timing, int do_print);

double boundarycondition_time=0.0, ghostcell_time=0.0;

int main(int argc, char *argv[])
{
   MPI_Init(&argc, &argv);

   int rank, nprocs;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   if (rank == 0) printf("------> Initializing the Problem\n");
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

   // physical domain dimensions (unit square)
   double Lx = 1.0;
   double Ly = 1.0;

   // soln init params
   double sigma = 5.0;
   double x_center = imax / 2.0;
   double y_center = jmax / 2.0;

   // center of the Gaussian for initialization
   double x0 = Lx / 2.0;
   double y0 = Ly / 2.0;

   int jstride = isize+2*nhalo;

   int jlow=0, jhgh=jsize;
   if (corners) {
      if (nbot == MPI_PROC_NULL) jlow = -nhalo;
      if (ntop == MPI_PROC_NULL) jhgh = jsize+nhalo;
   }
   int jnum = jhgh-jlow;
   int bufcount = jnum*nhalo;

   roctxRangePush("BufAlloc");
   std::vector<double> xbuf_left_send(bufcount);
   std::vector<double> xbuf_rght_send(bufcount);
   std::vector<double> xbuf_rght_recv(bufcount);
   std::vector<double> xbuf_left_recv(bufcount);
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
   init_core2 <<< grid2, block2 >>> (x, jmax, jspan, jbegin, jend, imax, ispan, ibegin, iend, sigma, jstride, nhalo);
   hipDeviceSynchronize();

   boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);

   if (do_print == 1) {
      if (rank == 0) printf("------> Initial State \n");
      // note: this ghostcell_update is only for printing purposes to show that the
      // solution matches the one produced by Orig also on the halo for debugging
      // the value of the solution on the halo does not matter as long as
      // it is up to date when it is used to compute the solution on cells that need it
      ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);
      HIP_CHECK(hipMemcpy(x_h, x, totcells * sizeof(double), hipMemcpyDefault));
      Cartesian_print(x_h, jmax, imax, nhalo, nprocy, nprocx, jstride);
   }

   roctxMark("Starting main iteration loop");
   if (rank == 0) printf("------> Advancing the Solution\n");

#ifdef USE_PNETCDF
   int ncid, varid, varid_xcoord, varid_ycoord;
     create_netcdf_file("solution.nc", jmax, imax, MPI_COMM_WORLD, &ncid, &varid, &varid_xcoord, &varid_ycoord);
     write_netcdf_soln(x, jmax, imax, nhalo, nprocy, nprocx, 0, ncid, varid, int jstride);
     write_netcdf_coords(imax, jmax, nprocx, nprocy, Lx, Ly, ncid, varid_xcoord, varid_ycoord);
#endif

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

      if (iter%10 == 0 && rank == 0) printf("        Iter %d\n",iter);
      if (do_print == 1) {
         HIP_CHECK(hipMemcpy(x_h, x, totcells * sizeof(double), hipMemcpyDefault));
         Cartesian_print(x_h, jmax, imax, nhalo, nprocy, nprocx, jstride);
      }
#ifdef USE_PNETCDF
      if(iter == maxIter - 1){
         write_netcdf_soln(x, jmax, imax, nhalo, nprocy, nprocx, iter+1, ncid, varid);
      }
#endif   
   }

   roctxMark("Stopping main iteration loop");

   total_time = cpu_timer_stop(tstart_total);

   if (rank == 0){
      printf("------> Printing Timings\n");
      printf("        Solution Advancement: %f \n", stencil_time);
      printf("        Boundary Condition Enforcement:  %f \n", boundarycondition_time);
      printf("        Ghost Cell Update:  %lf \n", ghostcell_time);
      printf("        Total: %f\n",total_time);
   }

   HIP_CHECK(hipFree(x));
   HIP_CHECK(hipFree(xnew));

   if (do_print) {
      free(x_h);
   }

   MPI_Finalize();
   exit(0);
}

void boundarycondition_update(double *x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop)
{
   int jstride = isize+2*nhalo;

   struct timespec tstart_boundarycondition;
   cpu_timer_start(&tstart_boundarycondition);

   dim3 gridVrt((nhalo+3)/4, (jsize+63)/64, 1);
   dim3 blockVrt(4, 64, 1);
   dim3 gridHor((isize+2*nhalo+63)/64, (nhalo+3)/4, 1);
   dim3 blockHor(64, 4, 1);

// Boundary conditions -- constant
   if (nleft == MPI_PROC_NULL){

      enforce_bcs_left <<< gridVrt, blockVrt >>> (x, nhalo, jsize, jstride);
      hipDeviceSynchronize();

   }

   if (nrght == MPI_PROC_NULL){

      enforce_bcs_rght <<< gridVrt, blockVrt >>> (x, nhalo, jsize, isize, jstride);
      hipDeviceSynchronize();

   }

   if (nbot == MPI_PROC_NULL){

      enforce_bcs_bot <<< gridHor, blockHor >>> (x, nhalo, isize, jstride);
      hipDeviceSynchronize();

   }
      
   if (ntop == MPI_PROC_NULL){

      enforce_bcs_top <<< gridHor, blockHor >>> (x, nhalo, jsize, isize, jstride);
      hipDeviceSynchronize();

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
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xbuf_left_send[offset] = xv(j,ll);
         }
      }
   }
   if (nrght != MPI_PROC_NULL){
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
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xv(j,isize+ll) = xbuf_rght_recv[offset];
         }
      }
   }
   if (nleft != MPI_PROC_NULL){
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
      int jmax, int imax, int nprocy, int nprocx, int do_timing, int do_print)
{
   int jstride = isize+2*nhalo;

   int rank;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   int totcells=(jsize+2*nhalo)*(isize+2*nhalo);
   double *x=NULL;
   HIP_CHECK(hipMalloc(&x, totcells*sizeof(double)));

   if (jsize > 12 || isize > 12) return;

   for (int j = -nhalo; j < jsize+nhalo; j++){
      for (int i = -nhalo; i < isize+nhalo; i++){
         xv(j,i) = 0.0;
      }
   }

   for (int j = 0; j < jsize; j++){
      for (int i = 0; i < isize; i++){
         xv(j,i) = rank * 1000 + j*10 + i;
      }
   }

   boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
   ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);

   if (do_print == 1) {
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

void Cartesian_print(double *x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int jstride)
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
            printf("%8d   ",i);
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
                  printf("%12.6lf ",xrow[i+ii*(isizes[ii]+2*nhalo)]);
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

#ifdef USE_PNETCDF

void create_netcdf_file(const char *fname, int jmax, int imax, MPI_Comm comm, int *ncid, int *varid, int *varid_xcoord, int *varid_ycoord)
{
    int dimid_t, dimid_y, dimid_x;
    int dimids[3];

    ncmpi_create(comm, fname, NC_CLOBBER | NC_64BIT_DATA, MPI_INFO_NULL, ncid);

    ncmpi_def_dim(*ncid, "time", NC_UNLIMITED, &dimid_t);
    ncmpi_def_dim(*ncid, "y", jmax, &dimid_y);
    ncmpi_def_dim(*ncid, "x", imax, &dimid_x);

    dimids[0] = dimid_t;
    dimids[1] = dimid_y;
    dimids[2] = dimid_x;

    ncmpi_def_var(*ncid, "u", NC_DOUBLE, 3, dimids, varid);
    ncmpi_def_var(*ncid, "xcoord", NC_DOUBLE, 1, &dimid_x, varid_xcoord);
    ncmpi_def_var(*ncid, "ycoord", NC_DOUBLE, 1, &dimid_y, varid_ycoord);

    ncmpi_put_att_text(*ncid, *varid, "u", 25, "solution field");
    ncmpi_put_att_text(*ncid, *varid_xcoord, "x", 11, "x coordinate");
    ncmpi_put_att_text(*ncid, *varid_ycoord, "y", 11, "y coordinate");

    ncmpi_enddef(*ncid);
}


void write_netcdf_soln(double *x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int tstep, int ncid, int varid, int jstride)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int xcoord = rank % nprocx;
    int ycoord = rank / nprocx;

    int ibegin = imax *  xcoord      / nprocx;
    int iend   = imax * (xcoord + 1) / nprocx;
    int isize  = iend - ibegin;

    int jbegin = jmax *  ycoord      / nprocy;
    int jend   = jmax * (ycoord + 1) / nprocy;
    int jsize  = jend - jbegin;

    MPI_Offset start[3], count[3];

    start[0] = tstep;
    start[1] = jmax - jend;
    start[2] = ibegin;

    count[0] = 1;
    count[1] = jsize;
    count[2] = isize;

    double *buf = (double *)malloc(jsize * isize * sizeof(double));

    for (int j = 0; j < jsize; j++) {
        for (int i = 0; i < isize; i++) {
            buf[(jsize-1-j)*isize + i] = xv(j,i);  // no halo cells included
        }
    }

    ncmpi_put_vara_double_all(ncid, varid, start, count, buf);

    free(buf);
}

void write_netcdf_coords(int imax, int jmax, int nprocx, int nprocy, double Lx, double Ly, int ncid, int varid_xcoord, int varid_ycoord)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int xcoord = rank % nprocx;
    int ycoord = rank / nprocx;

    int ibegin = imax *  xcoord      / nprocx;
    int iend   = imax * (xcoord + 1) / nprocx;
    int isize  = iend - ibegin;

    int jbegin = jmax *  ycoord      / nprocy;
    int jend   = jmax * (ycoord + 1) / nprocy;
    int jsize  = jend - jbegin;

    // x coordinates
    double *xbuf = (double *)malloc(isize * sizeof(double));
    for (int i = 0; i < isize; i++) {
        int iglobal = ibegin + i;
        xbuf[i] = (double)iglobal * Lx / imax;  // map to physical domain
    }

    MPI_Offset xstart = ibegin;
    MPI_Offset xcount = isize;

    ncmpi_put_vara_double_all(ncid, varid_xcoord, &xstart, &xcount, xbuf);

    free(xbuf);

    // y coordinates
    double *ybuf = (double *)malloc(jsize * sizeof(double));
    for (int j = 0; j < jsize; j++) {
        int jglobal = jbegin + j;
        ybuf[j] = (double)jglobal * Ly / jmax;  // map to physical domain
    }

    MPI_Offset ystart = jbegin;
    MPI_Offset ycount = jsize;

    ncmpi_put_vara_double_all(ncid, varid_ycoord, &ystart, &ycount, ybuf);

    free(ybuf);
}


void close_netcdf(int ncid)
{
    ncmpi_close(ncid);
}

#endif
