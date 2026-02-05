#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <mpi.h>
#include <omp.h>
#include <roctx.h>
#ifdef USE_PNETCDF
#include <pnetcdf.h>
#endif
#include <cmath>

#include "malloc2D.h"
#include "timer.h"

#pragma omp requires unified_shared_memory

#define SWAP_PTR(xnew,xold,xtmp) (xtmp=xnew, xnew=xold, xold=xtmp)
void parse_input_args(int argc, char **argv, int &jmax, int &imax, int &nprocy, int &nprocx, int &nhalo, int &corners, int &maxIter, int &do_timing, int &do_print);
void Cartesian_print(double **x, int jmax, int imax, int nhalo, int nprocy, int nprocx);
#ifdef USE_PNETCDF
void create_netcdf_file(const char *fname, int jmax, int imax, MPI_Comm comm, int *ncid, int *varid, int *varid_xcoord, int *varid_ycoord);
void write_netcdf_soln(double **x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int tstep, int ncid, int varid);
void write_netcdf_coords(int imax, int jmax, int nprocx, int nprocy, double Lx, double Ly, int ncid, int varid_xcoord, int varid_ycoord);
#endif
void boundarycondition_update(double **x, int nhalo, int jsize, int isize, int nleft, int nrght, int nbot, int ntop);
void ghostcell_update(double **x, int nhalo, int corners, int jsize, int isize,
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
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = -nhalo; j < jsize+nhalo; j++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            x[j][i] = 0.0;
         }
      }
   }

   #pragma omp target teams distribute parallel for collapse(2)
   for (int j = 0; j < jsize; j++) {
      for (int i = 0; i < isize; i++) {

        double x_phys = i + ibegin;
        double y_phys = j + jbegin;
        x[j][i] = exp(-0.5 * ( ((x_phys - x_center)*(x_phys - x_center)/(sigma*sigma)
                               + (y_phys - y_center)*(y_phys - y_center)/(sigma*sigma)) ));

       }
   }

   boundarycondition_update(x, nhalo, jsize, isize, nleft, nrght, nbot, ntop);
   ghostcell_update(x, nhalo, corners, jsize, isize, nleft, nrght, nbot, ntop, do_timing);

   if (do_print == 1) {
      if (rank == 0) printf("------> Initial State \n");
      Cartesian_print(x, jmax, imax, nhalo, nprocy, nprocx);
   }

   roctxMark("Starting main iteration loop");

   if (rank == 0) printf("------> Advancing the Solution\n");

#ifdef USE_PNETCDF
   int ncid, varid, varid_xcoord, varid_ycoord;
     create_netcdf_file("solution.nc", jmax, imax, MPI_COMM_WORLD, &ncid, &varid, &varid_xcoord, &varid_ycoord);
     write_netcdf_soln(x, jmax, imax, nhalo, nprocy, nprocx, 0, ncid, varid);
     write_netcdf_coords(imax, jmax, nprocx, nprocy, Lx, Ly, ncid, varid_xcoord, varid_ycoord);
#endif

   for (int iter = 0; iter < maxIter; iter++){
      roctxRangePush("Stencil");
      cpu_timer_start(&tstart_stencil);

      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = 0; j < jsize; j++){
         for (int i = 0; i < isize; i++){
            xnew[j][i] = ( x[j][i] + x[j][i-1] + x[j][i+1] + x[j-1][i] + x[j+1][i] )/5.0;
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

      if (iter%10 == 0 && rank == 0) printf("        Iter %d\n",iter);
      if (do_print == 1) {
         Cartesian_print(x, jmax, imax, nhalo, nprocy, nprocx);
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
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = 0; j < jsize; j++){
         for (int ll=-nhalo; ll<0; ll++){
            x[j][ll] = x[j][0];
         }
      }
   }

   if (nrght == MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = 0; j < jsize; j++){
         for (int ll=0; ll<nhalo; ll++){
            x[j][isize+ll] = x[j][isize-1];
         }
      }
   }

   if (nbot == MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int ll=-nhalo; ll<0; ll++){
         for (int i = -nhalo; i < isize+nhalo; i++){
            x[ll][i] = x[0][i];
         }
      }
   }
      
   if (ntop == MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
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

   roctxRangePush("BufAlloc");
   double *xbuf_left_send = (double *)omp_target_alloc(bufcount*sizeof(double), omp_get_default_device());
   double *xbuf_rght_send = (double *)omp_target_alloc(bufcount*sizeof(double), omp_get_default_device());
   double *xbuf_rght_recv = (double *)omp_target_alloc(bufcount*sizeof(double), omp_get_default_device());
   double *xbuf_left_recv = (double *)omp_target_alloc(bufcount*sizeof(double), omp_get_default_device());
   int offset = 0;
   roctxRangePop(); //BufAlloc

   roctxRangePush("LoadLeftRight");
   if (nleft != MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xbuf_left_send[offset] = x[j][ll];
         }
      }
   }
   if (nrght != MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            xbuf_rght_send[offset] = x[j][isize-nhalo+ll];
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
            x[j][isize+ll] = xbuf_rght_recv[offset];
         }
      }
   }
   if (nleft != MPI_PROC_NULL){
      #pragma omp target teams distribute parallel for collapse(2)
      for (int j = jlow; j < jhgh; j++){
         for (int ll = 0; ll < nhalo; ll++){
            offset = (j - jlow) * nhalo + ll;
            x[j][-nhalo+ll] = xbuf_left_recv[offset];
         }
      }
   }
   roctxRangePop(); //UnpackLeftRight

   roctxRangePush("BufFree");
   omp_target_free(xbuf_left_send, omp_get_default_device());
   omp_target_free(xbuf_rght_send, omp_get_default_device());
   omp_target_free(xbuf_rght_recv, omp_get_default_device());
   omp_target_free(xbuf_left_recv, omp_get_default_device());
   roctxRangePop(); //BufFree

   roctxRangePush("MPIUpDownExchange");
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
   roctxRangePop(); //MPIUpDownExchange

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

   #pragma omp target teams distribute parallel for collapse(2)
   for (int j = -nhalo; j < jsize+nhalo; j++){
      for (int i = -nhalo; i < isize+nhalo; i++){
         x[j][i] = 0.0;
      }
   }

   #pragma omp target teams distribute parallel for collapse(2)
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
            printf("%9d   ",i);
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


void write_netcdf_soln(double **x, int jmax, int imax, int nhalo, int nprocy, int nprocx, int tstep, int ncid, int varid)
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
            buf[(jsize-1-j)*isize + i] = x[j][i];  // no halo cells included
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
