#!/bin/bash

# This test compiles and runs a standalone ELPA real-double 1-stage AMD GPU
# eigenvectors test against the installed ELPA library, on 2 MPI ranks.
# It uses a symmetrized Clement (symmetric tridiagonal) matrix whose
# eigenvalues are known analytically -- they are the integers -na+1,
# -na+3, ..., na-3, na-1 -- and verifies the computed eigenvalues against
# them. See https://www.netlib.org/lapack/lawnspdf/lawn182.pdf, Sec. 2.2.1
# for the matrix definition, and the ELPA tutorial example
# https://github.com/karpov-peter/elpa-tutorial/blob/main/examples_Fortran/04_eigenproblem_elpa/eigenproblem_elpa.f90
# for the same construction in Fortran.

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
   integer :: success, argc, i, pass_flag, n_print
   integer :: i_loc, j_loc, l_1, x_1, l_2, x_2, I_gl, J_gl, k_min
   integer, external :: numroc

   real(kind=c_double), allocatable :: a(:,:), z(:,:), ev(:)
   real(kind=c_double), allocatable :: ev_exact(:), ev_diff(:)

   real(kind=c_double) :: t0, t1, max_abs_err, tol
   character(len=64)   :: argbuf
   class(elpa_t), pointer :: e

   ! defaults: keep the upstream validate_* matrix size
   na   = 20000
   nev  = 20000
   nblk = 64

   ! Tolerance for the eigenvalue correctness check. Expected error is
   ! O(machine_epsilon * |lambda_max|) ~ 5e-12 for na=20000; we leave a
   ! safety margin to absorb GPU floating-point quirks across solvers.
   tol = 1.0e-8_c_double

   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world, myid,   mpierr)
   call mpi_comm_size(mpi_comm_world, nprocs, mpierr)

   ! argv: na nev nblk
   argc = command_argument_count()
   if (argc >= 1) then; call get_command_argument(1, argbuf); read(argbuf,*) na;   end if
   if (argc >= 2) then; call get_command_argument(2, argbuf); read(argbuf,*) nev;  end if
   if (argc >= 3) then; call get_command_argument(3, argbuf); read(argbuf,*) nblk; end if

   if (myid == 0) then
      print '(a)',              '=========================================='
      print '(a,i0)',           ' ELPA real-double 1stage GPU test  ranks=', nprocs
      print '(a,i0,a,i0,a,i0)', ' na=', na, '  nev=', nev, '  nblk=', nblk
      print '(a)',              ' matrix: symmetrized Clement (analytic eigenvalues)'
      print '(a)',              '=========================================='
   end if

   ! Near-square BLACS grid (np_rows * np_cols == nprocs).
   np_cols = 1
   do i = nint(sqrt(real(nprocs))), 1, -1
      if (mod(nprocs, i) == 0) then
         np_cols = i
         exit
      end if
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

   allocate(a (na_rows, na_cols))
   allocate(z (na_rows, na_cols))
   allocate(ev(na))

   ! Fill A locally with the symmetrized Clement matrix:
   !   A is symmetric tridiagonal with zero diagonal and
   !   A(i, i+1) = A(i+1, i) = sqrt(i * (na - i)) for i = 1..na-1.
   ! Eigenvalues are exactly -na+1, -na+3, ..., na-3, na-1.
   ! The (i_loc, j_loc) -> (I_gl, J_gl) mapping is the standard ScaLAPACK
   ! block-cyclic one for an (na x na) matrix on an (np_rows x np_cols)
   ! BLACS grid with block size nblk and rsrc=csrc=0.
   a(:,:) = 0.0_c_double
   do j_loc = 1, na_cols
      l_2 = (j_loc - 1) / nblk
      x_2 = mod(j_loc - 1, nblk)
      J_gl = (l_2 * np_cols + my_pcol) * nblk + x_2 + 1
      if (J_gl > na) cycle
      do i_loc = 1, na_rows
         l_1 = (i_loc - 1) / nblk
         x_1 = mod(i_loc - 1, nblk)
         I_gl = (l_1 * np_rows + my_prow) * nblk + x_1 + 1
         if (I_gl > na) cycle
         if (abs(I_gl - J_gl) == 1) then
            k_min = min(I_gl, J_gl)
            a(i_loc, j_loc) = sqrt(real(k_min, c_double) * real(na - k_min, c_double))
         end if
      end do
   end do

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

   ! Correctness check on rank 0: ELPA returns the smallest nev eigenvalues
   ! in ascending order, so ev(i) should equal -na+1+2*(i-1) for i=1..nev.
   pass_flag   = 1
   max_abs_err = 0.0_c_double
   if (myid == 0) then
      allocate(ev_exact(nev), ev_diff(nev))
      do i = 1, nev
         ev_exact(i) = real(-na + 1 + 2*(i-1), c_double)
         ev_diff(i)  = ev(i) - ev_exact(i)
      end do
      max_abs_err = maxval(abs(ev_diff))

      print '(a,f12.3,a)', ' eigenvectors() wall time : ', t1 - t0, ' s'
      print '(a)',         ' sample eigenvalues vs analytic (lambda_i = -na+1+2*(i-1)):'
      n_print = min(5, nev)
      do i = 1, n_print
         print '(a,i6,a,es23.16,a,es23.16,a,es10.3)', &
            '   i=', i, '  computed=', ev(i), '  analytic=', ev_exact(i), &
            '  abs diff=', abs(ev_diff(i))
      end do
      if (nev > 2 * n_print) then
         print '(a)', '   ...'
         do i = nev - n_print + 1, nev
            print '(a,i6,a,es23.16,a,es23.16,a,es10.3)', &
               '   i=', i, '  computed=', ev(i), '  analytic=', ev_exact(i), &
               '  abs diff=', abs(ev_diff(i))
         end do
      end if
      print '(a,i0,a,es12.5)', ' max |computed - analytic| over all ', nev, &
                               ' eigenvalues : ', max_abs_err
      print '(a,es12.5)',      ' tolerance                                                 : ', tol
      if (max_abs_err <= tol) then
         print '(a)', ' RESULT: PASS'
      else
         print '(a)', ' RESULT: FAIL'
         pass_flag = 0
      end if

      deallocate(ev_exact, ev_diff)
   end if

   ! Propagate the pass/fail decision to all ranks so the program exits
   ! consistently (and mpirun returns non-zero on failure).
   call mpi_bcast(pass_flag, 1, mpi_integer, 0, mpi_comm_world, mpierr)

   call elpa_deallocate(e)
   call elpa_uninit()
   call blacs_gridexit(my_blacs_ctxt)
   call mpi_finalize(mpierr)

   if (pass_flag /= 1) call exit(1)
end program elpa_test_real_gpu
EOF

pushd "$WORKDIR"

ELPA_CFLAGS="$(pkg-config --cflags elpa)"
ELPA_LIBS="$(pkg-config --libs elpa)"

mpifort -O3 elpa_test_real_gpu.F90 ${ELPA_CFLAGS} -L${ROCM_PATH}/lib ${ELPA_LIBS} -I${ELPA_PATH}/include/elpa-2026.02.001/modules -o elpa_test_real_gpu

mpirun -np 2 ./elpa_test_real_gpu 20000 20000 64

popd
