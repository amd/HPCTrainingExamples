#!/bin/bash

# This test compiles and runs a standalone ELPA real-double 1-stage AMD GPU
# eigenvectors test against the installed ELPA library. It mirrors the
# upstream validate_real_double_eigenvectors_1stage_gpu_api_random_explicit
# test: random symmetric matrix, 1-stage solver, AMD GPU, correctness check
# skipped (we just time the eigenvectors call).

# NOTE: this test assumes ELPA has been installed according
# to the instructions available in the model installation repo:
# https://github.com/amd/HPCTrainingDock/blob/main/extras/scripts/elpa_setup.sh


module -t list 2>&1 | grep -q "^rocm"
if [ $? -eq 1 ]; then
  echo "rocm module is not loaded"
  echo "loading default rocm module"
  module load rocm
fi
# The elpa module loads petsc, which in turn loads the MPI module
# (openmpi by default), so we get scalapack/BLAS via PETSC_PATH and
# mpifort/mpirun on PATH transitively.
module load elpa
module list

ulimit -s unlimited

if ! pkg-config --exists elpa; then
   echo "ERROR: 'pkg-config --exists elpa' failed -- did you 'module load elpa/<version>'?" >&2
   exit 1
fi

WORKDIR=$(mktemp -d -t elpa_test_XXXXXXXXXX)
trap "rm -rf $WORKDIR" EXIT

cat > "$WORKDIR/elpa_test_real_gpu.F90" << 'EOF'
program elpa_test_real_gpu
   use iso_c_binding
   use elpa
#ifdef HAVE_MPI_MODULE
   use mpi
   implicit none
#else
   implicit none
   include 'mpif.h'
#endif

   integer :: na, nev, nblk
   integer :: np_rows, np_cols, na_rows, na_cols
   integer :: myid, nprocs, my_prow, my_pcol
   integer :: mpierr, my_blacs_ctxt, sc_desc(9), info, nprow, npcol
   integer :: success, argc, i
   integer, external :: numroc

   real(kind=c_double), allocatable :: a(:,:), z(:,:), ev(:)
   integer                          :: iseed(4096)

   real(kind=c_double) :: t0, t1
   character(len=64)   :: argbuf
   class(elpa_t), pointer :: e

   ! defaults match the upstream validate_* invocation's first three args
   na   = 40000
   nev  = 40000
   nblk = 64

   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world, myid,   mpierr)
   call mpi_comm_size(mpi_comm_world, nprocs, mpierr)

   ! argv: na nev nblk [skip_check_correctness]
   ! arg 4 is accepted for CLI parity with the upstream validate_* tests
   ! but ignored: this program never checks correctness.
   argc = command_argument_count()
   if (argc >= 1) then; call get_command_argument(1, argbuf); read(argbuf,*) na;   end if
   if (argc >= 2) then; call get_command_argument(2, argbuf); read(argbuf,*) nev;  end if
   if (argc >= 3) then; call get_command_argument(3, argbuf); read(argbuf,*) nblk; end if

   if (myid == 0) then
      print '(a)',         '=========================================='
      print '(a,i0,a,i0)', ' ELPA real-double 1stage GPU test  ranks=', nprocs
      print '(a,i0,a,i0,a,i0)', ' na=', na, '  nev=', nev, '  nblk=', nblk
      print '(a)',         '=========================================='
   end if

   ! near-square BLACS grid
   do np_cols = nint(sqrt(real(nprocs))), 2, -1
      if (mod(nprocs, np_cols) == 0) exit
   end do
   np_rows = nprocs / np_cols

   my_blacs_ctxt = mpi_comm_world
   call BLACS_Gridinit(my_blacs_ctxt, 'C', np_rows, np_cols)
   call BLACS_Gridinfo(my_blacs_ctxt, nprow, npcol, my_prow, my_pcol)

   na_rows = numroc(na, nblk, my_prow, 0, np_rows)
   na_cols = numroc(na, nblk, my_pcol, 0, np_cols)
   call descinit(sc_desc, na, na, nblk, nblk, 0, 0, my_blacs_ctxt, na_rows, info)
   if (info /= 0) then
      if (myid == 0) print *, 'descinit failed, info=', info, &
         ' -- too many MPI tasks for this matrix/blocksize?'
      call MPI_ABORT(mpi_comm_world, 1, mpierr)
   end if

   ! random symmetric matrix
   allocate(a (na_rows, na_cols))
   allocate(z (na_rows, na_cols))
   allocate(ev(na))

   iseed(:) = myid
   call RANDOM_SEED(put = iseed)
   call RANDOM_NUMBER(z)
   a(:,:) = z(:,:)
   call pdtran(na, na, 1.0d0, z, 1, 1, sc_desc, 1.0d0, a, 1, 1, sc_desc) ! A := A + Z**T

   if (elpa_init(20171201) /= elpa_ok) then
      if (myid == 0) print *, 'ELPA API version not supported'
      call MPI_ABORT(mpi_comm_world, 2, mpierr)
   end if
   e => elpa_allocate()

   call e%set("na",              na,              success)
   call e%set("nev",             nev,             success)
   call e%set("local_nrows",     na_rows,         success)
   call e%set("local_ncols",     na_cols,         success)
   call e%set("nblk",            nblk,            success)
   call e%set("mpi_comm_parent", mpi_comm_world,  success)
   call e%set("process_row",     my_prow,         success)
   call e%set("process_col",     my_pcol,         success)

   success = e%setup()

   call e%set("solver",  elpa_solver_1stage, success)
   call e%set("amd-gpu", 1,                  success)

   call mpi_barrier(mpi_comm_world, mpierr)
   t0 = MPI_Wtime()
   call e%eigenvectors(a, ev, z, success)
   call mpi_barrier(mpi_comm_world, mpierr)
   t1 = MPI_Wtime()

   if (myid == 0) then
      print '(a,f12.3,a)', ' eigenvectors() wall time: ', t1 - t0, ' s'
      print '(a,es14.6)',  ' first eigenvalue       : ', ev(1)
      print '(a,es14.6)',  ' last  eigenvalue       : ', ev(min(nev, na))
   end if

   call elpa_deallocate(e)
   call elpa_uninit()
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)
end program elpa_test_real_gpu
EOF

pushd "$WORKDIR"

ELPA_CFLAGS="$(pkg-config --cflags elpa)"
ELPA_LIBS="$(pkg-config --libs elpa)"

mpifort -O3 elpa_test_real_gpu.F90 ${ELPA_CFLAGS} -L${ROCM_PATH}/lib ${ELPA_LIBS} -I${ELPA_PATH}/include/elpa-2026.02.001/modules -o elpa_test_real_gpu

mpirun -np 1 ./elpa_test_real_gpu 20000 20000 64 skip_check_correctness

popd
