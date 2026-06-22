#!/bin/bash

# Checking for errors when UCC collectives produce seg faults with OpenMP
# when messages are passed across multiple GPUs 
#
# Three device-memory provisioning paths are used:
#   - use_device_addr : OpenMP-offload mapped Fortran allocatables passed to
#                       MPI inside !$omp target data use_device_addr (example.F90)
#   - omp_target_alloc: raw device pointer from omp_target_alloc, no mapping
#                       table entry, handed straight to MPI (example_alloc.F90)
#   - alloc+addr (mix): omp_target_alloc memory (~ hipMalloc) bound via
#                       c_f_pointer, then ALSO put through map(to:)
#                       use_device_addr around the MPI call (example_mix.F90)
#
# With UCC collectives ON and OFF, the above gives 6 cases total. On affected UCC
# stacks the UCC-ON cases fault with a GPU memory access fault (exit 134);
# --mca coll_ucc_enable 0 is the known workaround. A healthy stack passes all
# 6. REGRESSION_RESULT: PASS iff all 6 pass.
#
# REQUIREMENTS to trigger the fault:
#   * the two ranks must land on TWO DIFFERENT GPU devices
#     (omp_set_default_device(local_rank)); same-device runs do NOT fault.
#       - CPX: a single APU exposes multiple XCD devices -> one APU is enough.
#       - SPX: one APU == one device -> needs 2 APUs visible.
#   * HSA_XNACK=0 (set below). HSA_XNACK=1 masks the fault via USM.
#
# Self-contained: all sources are generated and built here.

if [[ -n "$CRAYPE_VERSION" || -f /etc/cray-release ]]; then
   if [ -z "$CXX" ]; then
      export CXX=`which CC`
   fi
   if [ -z "$CC" ]; then
      export CC=`which cc`
   fi
   if [ -z "$FC" ]; then
      export FC=`which ftn`
   fi
else
   module -t list 2>&1 | grep -q "^rocm"
   if [ $? -eq 1 ]; then
     echo "rocm module is not loaded"
     echo "loading default rocm module"
     module load rocm
   fi
   module load amdflang-new >& /dev/null
   if [ "$?" == "1" ]; then
      module load amdclang
   fi
   module load openmpi
fi

# Need >= 2 visible GPU devices so the 2 ranks land on different devices.
GPU_COUNT=`rocminfo | grep "Device Type:             GPU" | wc -l`
if [ ${GPU_COUNT} -lt 2 ]; then
   echo "Skip"
else

   GFX_MODEL=`rocminfo | grep gfx | sed -e 's/Name://' | head -1 | sed 's/ //g'`
   [ -z "${GFX_MODEL}" ] && GFX_MODEL=gfx942

   WORKDIR=`mktemp -d`
   trap "rm -rf ${WORKDIR}" EXIT

   cat > ${WORKDIR}/example.F90 <<'EOF'
module example_buffers
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none
  integer, parameter :: rp = real64
  real(rp), allocatable, dimension(:) :: sbuf, rbuf
end module example_buffers

module example_kernels
  use example_buffers, only : rp
  implicit none
contains
  subroutine reduce_dev(n, a0, a1, comm, ierr)
    use mpi
    implicit none
    integer, intent(in)    :: n, comm
    integer, intent(out)   :: ierr
    real(rp)               :: a0(n), a1(n)
!$omp target data use_device_addr(a0,a1)
    call MPI_Allreduce(MPI_IN_PLACE, a0, n, MPI_REAL8, MPI_SUM, comm, ierr)
    call MPI_Allreduce(MPI_IN_PLACE, a1, n, MPI_REAL8, MPI_SUM, comm, ierr)
!$omp end target data
  end subroutine reduce_dev
end module example_kernels

program example
  use mpi
  use omp_lib
  use example_buffers
  use example_kernels
  implicit none

  integer :: ierr, nrank, myrank, shmcomm, local_rank, ndev, i, n
  integer :: nbr_up, nbr_dn, reqs(2)
  real(rp), allocatable, dimension(:) :: sum0, sum1
  real(rp) :: expect_sum, expect_nbr
  logical  :: ok
  integer  :: tag = 100

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nrank, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)

  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
                           MPI_INFO_NULL, shmcomm, ierr)
  call MPI_Comm_rank(shmcomm, local_rank, ierr)

  ndev = omp_get_num_devices()
!$ call omp_set_default_device(local_rank)
  write(*,'(a,i0,a,i0,a,i0,a,i0)') 'rank ', myrank, ' local ', local_rank, &
       ' default_device ', omp_get_default_device(), ' ndev ', ndev
  flush(6)

  n = 4096

  !---------------------------------------------------------------------------------------
  ! TEST A: dummy-arg use_device_addr + MPI_Allreduce(MPI_IN_PLACE) this is the UCC track
  !---------------------------------------------------------------------------------------
  allocate(sum0(n), sum1(n))
  sum0 = real(myrank + 1, rp)
  sum1 = real(2*(myrank + 1), rp)
!$omp target enter data map(to:sum0,sum1)
!$omp target teams distribute parallel do
  do i = 1, n
     sum0(i) = sum0(i) + 0.0_rp
     sum1(i) = sum1(i) + 0.0_rp
  end do

  call reduce_dev(n, sum0, sum1, MPI_COMM_WORLD, ierr)

!$omp target update from(sum0,sum1)
!$omp target exit data map(delete:sum0,sum1)

  expect_sum = real(nrank*(nrank+1)/2, rp)
  ok = all(abs(sum0 - expect_sum)        < 1.0e-9_rp) .and. &
       all(abs(sum1 - 2.0_rp*expect_sum) < 1.0e-9_rp)
  call report('TEST_A_allreduce_use_device_addr', ok, myrank)
  deallocate(sum0, sum1)

  !------------------------------------------------------------------------------------------------
  ! TEST B: module-buffer use_device_addr + MPI_Isend/Irecv this is a pt2pt just as a sanity check
  !------------------------------------------------------------------------------------------------
  allocate(sbuf(n), rbuf(n))
  sbuf = real(myrank + 1, rp)
  rbuf = -1.0_rp
!$omp target enter data map(to:sbuf,rbuf)
!$omp target teams distribute parallel do
  do i = 1, n
     sbuf(i) = sbuf(i) + 0.0_rp
  end do

  nbr_up = modulo(myrank + 1, nrank)
  nbr_dn = modulo(myrank - 1, nrank)

!$omp target data use_device_addr(sbuf,rbuf)
  call MPI_Irecv(rbuf, n, MPI_REAL8, nbr_dn, tag, MPI_COMM_WORLD, reqs(1), ierr)
  call MPI_Isend(sbuf, n, MPI_REAL8, nbr_up, tag, MPI_COMM_WORLD, reqs(2), ierr)
  call MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE, ierr)
!$omp end target data

!$omp target update from(rbuf)
!$omp target exit data map(delete:sbuf,rbuf)

  expect_nbr = real(nbr_dn + 1, rp)
  ok = all(abs(rbuf - expect_nbr) < 1.0e-9_rp)
  call report('TEST_B_sendrecv_use_device_addr', ok, myrank)
  deallocate(sbuf, rbuf)

  call MPI_Finalize(ierr)

contains
  subroutine report(name, ok, myrank)
    character(*), intent(in) :: name
    logical,      intent(in) :: ok
    integer,      intent(in) :: myrank
    if (ok) then
      write(*,'(a,i0,a,a)') 'rank ', myrank, ' RESULT: PASS  ', name
    else
      write(*,'(a,i0,a,a)') 'rank ', myrank, ' RESULT: FAIL  ', name
    end if
    flush(6)
  end subroutine report
end program example
EOF

   cat > ${WORKDIR}/example_alloc.F90 <<'EOF'
! Same two collectives as example.F90, but on RAW device memory obtained from
! omp_target_alloc -- no map(), no use_device_addr, not in the OpenMP mapping
! table. The c_f_pointer-backed array (device address) is handed straight to
! GPU-aware MPI. Same language/compiler as example.F90, so the ONLY difference
! between the two tracks is the device-memory provisioning path.
program example_alloc
  use mpi
  use omp_lib
  use iso_c_binding
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none
  integer, parameter :: rp = real64

  integer :: ierr, nrank, myrank, shmcomm, local_rank, ndev, n, dev, host
  integer :: nbr_up, nbr_dn, reqs(2), rc
  integer :: tag = 100
  integer(c_size_t)         :: nbytes
  type(c_ptr)               :: csend, crecv
  real(rp), pointer         :: dsend(:), drecv(:)
  real(rp), allocatable, target :: h(:)
  real(rp) :: expect_sum, expect_nbr
  logical  :: ok

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nrank, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
                           MPI_INFO_NULL, shmcomm, ierr)
  call MPI_Comm_rank(shmcomm, local_rank, ierr)

  ndev = omp_get_num_devices()
  call omp_set_default_device(local_rank)
  dev  = omp_get_default_device()
  host = omp_get_initial_device()
  write(*,'(a,i0,a,i0,a,i0,a,i0)') 'rank ', myrank, ' local ', local_rank, &
       ' default_device ', dev, ' ndev ', ndev
  flush(6)

  n = 4096
  nbytes = int(n, c_size_t) * 8_c_size_t        ! real64 = 8 bytes
  allocate(h(n))

  ! Raw device allocations (not in the OpenMP mapping table).
  csend = omp_target_alloc(nbytes, dev)
  crecv = omp_target_alloc(nbytes, dev)
  call c_f_pointer(csend, dsend, [n])
  call c_f_pointer(crecv, drecv, [n])

  ! TEST A: in-place allreduce on the raw device pointer.
  h = real(myrank + 1, rp)
  rc = omp_target_memcpy(csend, c_loc(h), nbytes, 0_c_size_t, 0_c_size_t, dev, host)
  call MPI_Allreduce(MPI_IN_PLACE, dsend, n, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
  rc = omp_target_memcpy(c_loc(h), csend, nbytes, 0_c_size_t, 0_c_size_t, host, dev)
  expect_sum = real(nrank*(nrank+1)/2, rp)
  ok = all(abs(h - expect_sum) < 1.0e-9_rp)
  call report('TEST_A_allreduce_omp_target_alloc', ok, myrank)

  ! TEST B: Isend/Irecv on raw device pointers.
  h = real(myrank + 1, rp)
  rc = omp_target_memcpy(csend, c_loc(h), nbytes, 0_c_size_t, 0_c_size_t, dev, host)
  nbr_up = modulo(myrank + 1, nrank)
  nbr_dn = modulo(myrank - 1, nrank)
  call MPI_Irecv(drecv, n, MPI_REAL8, nbr_dn, tag, MPI_COMM_WORLD, reqs(1), ierr)
  call MPI_Isend(dsend, n, MPI_REAL8, nbr_up, tag, MPI_COMM_WORLD, reqs(2), ierr)
  call MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE, ierr)
  rc = omp_target_memcpy(c_loc(h), crecv, nbytes, 0_c_size_t, 0_c_size_t, host, dev)
  expect_nbr = real(nbr_dn + 1, rp)
  ok = all(abs(h - expect_nbr) < 1.0e-9_rp)
  call report('TEST_B_sendrecv_omp_target_alloc', ok, myrank)

  call omp_target_free(csend, dev)
  call omp_target_free(crecv, dev)
  call MPI_Finalize(ierr)

contains
  subroutine report(name, ok, myrank)
    character(*), intent(in) :: name
    logical,      intent(in) :: ok
    integer,      intent(in) :: myrank
    if (ok) then
      write(*,'(a,i0,a,a)') 'rank ', myrank, ' RESULT: PASS  ', name
    else
      write(*,'(a,i0,a,a)') 'rank ', myrank, ' RESULT: FAIL  ', name
    end if
    flush(6)
  end subroutine report
end program example_alloc
EOF

   cat > ${WORKDIR}/example_mix.F90 <<'EOF'
! MIX of the two paths: memory is obtained from omp_target_alloc (on AMD GPUs
! effectively a hipMalloc, OUTSIDE any target region) and bound to a Fortran
! pointer via c_f_pointer -- but it is then ALSO handed to OpenMP through
! !$omp target data map(to:...) use_device_addr(...) around the MPI call.
program example_mix
  use mpi
  use omp_lib
  use iso_c_binding
  use, intrinsic :: iso_fortran_env, only : real64
  implicit none
  integer, parameter :: rp = real64

  integer :: ierr, nrank, myrank, shmcomm, local_rank, ndev, n, dev, host
  integer :: nbr_up, nbr_dn, reqs(2), rc
  integer :: tag = 100
  integer(c_size_t)         :: nbytes
  type(c_ptr)               :: d_send, d_recv
  real(rp), pointer         :: sbuf(:), rbuf(:)
  real(rp), allocatable, target :: h(:)
  real(rp) :: expect_sum, expect_nbr
  logical  :: ok

  call MPI_Init(ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nrank, ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, myrank, ierr)
  call MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, &
                           MPI_INFO_NULL, shmcomm, ierr)
  call MPI_Comm_rank(shmcomm, local_rank, ierr)

  ndev = omp_get_num_devices()
  call omp_set_default_device(local_rank)
  dev  = omp_get_default_device()
  host = omp_get_initial_device()
  write(*,'(a,i0,a,i0,a,i0,a,i0)') 'rank ', myrank, ' local ', local_rank, &
       ' default_device ', dev, ' ndev ', ndev
  flush(6)

  n = 4096
  nbytes = int(n, c_size_t) * 8_c_size_t        ! real64 = 8 bytes
  allocate(h(n))

  ! omp_target_alloc OUTSIDE a target region (~ hipMalloc), bind Fortran ptr.
  d_send = omp_target_alloc(nbytes, dev)
  d_recv = omp_target_alloc(nbytes, dev)
  call c_f_pointer(d_send, sbuf, [n])
  call c_f_pointer(d_recv, rbuf, [n])

  ! TEST A: in-place allreduce. map(to:) brings the alloc'd buffer into the
  ! device data environment; use_device_addr hands its device address to MPI;
  ! target update from copies the reduced values back into the same buffer
  ! before we read it (avoids the decoupled second-buffer pitfall).
  h = real(myrank + 1, rp)
  rc = omp_target_memcpy(d_send, c_loc(h), nbytes, 0_c_size_t, 0_c_size_t, dev, host)
!$omp target enter data map(to:sbuf)
!$omp target data use_device_addr(sbuf)
  call MPI_Allreduce(MPI_IN_PLACE, sbuf, n, MPI_REAL8, MPI_SUM, MPI_COMM_WORLD, ierr)
!$omp end target data
!$omp target update from(sbuf)
!$omp target exit data map(delete:sbuf)
  rc = omp_target_memcpy(c_loc(h), d_send, nbytes, 0_c_size_t, 0_c_size_t, host, dev)
  expect_sum = real(nrank*(nrank+1)/2, rp)
  ok = all(abs(h - expect_sum) < 1.0e-9_rp)
  call report('TEST_A_allreduce_alloc_use_device_addr', ok, myrank)

  ! TEST B: Isend/Irecv. Same map / use_device_addr / update-from structure.
  h = real(myrank + 1, rp)
  rc = omp_target_memcpy(d_send, c_loc(h), nbytes, 0_c_size_t, 0_c_size_t, dev, host)
  nbr_up = modulo(myrank + 1, nrank)
  nbr_dn = modulo(myrank - 1, nrank)
!$omp target enter data map(to:sbuf,rbuf)
!$omp target data use_device_addr(sbuf,rbuf)
  call MPI_Irecv(rbuf, n, MPI_REAL8, nbr_dn, tag, MPI_COMM_WORLD, reqs(1), ierr)
  call MPI_Isend(sbuf, n, MPI_REAL8, nbr_up, tag, MPI_COMM_WORLD, reqs(2), ierr)
  call MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE, ierr)
!$omp end target data
!$omp target update from(rbuf)
!$omp target exit data map(delete:sbuf,rbuf)
  rc = omp_target_memcpy(c_loc(h), d_recv, nbytes, 0_c_size_t, 0_c_size_t, host, dev)
  expect_nbr = real(nbr_dn + 1, rp)
  ok = all(abs(h - expect_nbr) < 1.0e-9_rp)
  call report('TEST_B_sendrecv_alloc_use_device_addr', ok, myrank)

  call omp_target_free(d_send, dev)
  call omp_target_free(d_recv, dev)
  call MPI_Finalize(ierr)

contains
  subroutine report(name, ok, myrank)
    character(*), intent(in) :: name
    logical,      intent(in) :: ok
    integer,      intent(in) :: myrank
    if (ok) then
      write(*,'(a,i0,a,a)') 'rank ', myrank, ' RESULT: PASS  ', name
    else
      write(*,'(a,i0,a,a)') 'rank ', myrank, ' RESULT: FAIL  ', name
    end if
    flush(6)
  end subroutine report
end program example_mix
EOF

   if [ -n "${CRAY_MPICH_VERSION:-}" ]; then
      MPIFORT=ftn
      OPENMP_FLAGS=-fopenmp
      MPIRUN=srun
      MPIRUN_OPTIONS=
   else
      MPIFORT=mpifort
      OPENMP_FLAGS=-fopenmp --offload-arch=${GFX_MODEL}
      MPIRUN=mpirun
      MPIRUN_OPTIONS="--bind-to none -mca pml ucx -mca coll_ucc_enable"
   fi

   cd ${WORKDIR}
   ${MPIFORT} -O3 ${OPENMP_FLAGS} example.F90       -o example_addr
   ${MPIFORT} -O3 ${OPENMP_FLAGS} example_alloc.F90 -o example_alloc
   ${MPIFORT} -O3 ${OPENMP_FLAGS} example_mix.F90   -o example_mix

   # HSA_XNACK=0 exposes the fault (HSA_XNACK=1 masks it via USM).
   export HSA_XNACK=0
   export OMP_PROC_BIND=spread
   export OMP_PLACES=cores
   ulimit -s unlimited

   # Run one case: <label> <binary> <ucc_enable 0|1>.
   # A case passes when it exits 0 with 4 "RESULT: PASS" lines and no FAIL.
   ncase_fail=0
   run_case() {
      label="$1"; bin="$2"; ucc="$3"
      echo "=================================================================="
      echo "  CASE ${label}  (coll_ucc_enable=${ucc})"
      echo "=================================================================="
      out=`${MPIRUN} -np 2 ${MPIRUN_OPTIONS} ${ucc} ./${bin} 2>&1`
      rc=$?
      echo "${out}"
      npass=`echo "${out}" | grep -c 'RESULT: PASS'`
      nfail=`echo "${out}" | grep -c 'RESULT: FAIL'`
      if [ ${rc} -eq 0 ] && [ ${npass} -ge 4 ] && [ ${nfail} -eq 0 ]; then
         echo "CASE_RESULT ${label}: PASS (exit=${rc} pass=${npass} fail=${nfail})"
      else
         echo "CASE_RESULT ${label}: FAIL (exit=${rc} pass=${npass} fail=${nfail})"
         ncase_fail=`expr ${ncase_fail} + 1`
      fi
   }

   run_case "use_device_addr__UCC_on"          example_addr  1
   run_case "use_device_addr__UCC_off"         example_addr  0
   run_case "omp_target_alloc__UCC_on"         example_alloc 1
   run_case "omp_target_alloc__UCC_off"        example_alloc 0
   run_case "alloc_plus_use_device_addr__UCC_on"  example_mix 1
   run_case "alloc_plus_use_device_addr__UCC_off" example_mix 0

   echo "=================================================================="
   if [ ${ncase_fail} -eq 0 ]; then
      echo "REGRESSION_RESULT: PASS (all 6 cases passed)"
   else
      echo "REGRESSION_RESULT: FAIL (${ncase_fail} of 6 cases faulted)"
   fi
fi
