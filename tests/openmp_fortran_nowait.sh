#!/bin/bash

if [[ "`printenv |grep -w CRAY |wc -l`" -gt 1 ]]; then
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
fi

# Inline regression test for `target ... nowait`.
#
# The user-facing nowait example lives in
# Pragma_Examples/OpenMP/Fortran/Nowait/. This test is a self-contained
# single-kernel timing harness that verifies the underlying property
# the multi-kernel pattern relies on: when `target ... nowait` is
# placed inside an OpenMP parallel region, the encountering thread
# returns from the kernel launch in microseconds rather than waiting
# for the GPU kernel to complete.

FC=${FC:-amdflang}
ROCM_GPU=$(rocminfo 2>/dev/null | grep -m 1 -E 'gfx[^0]{1}' | sed -e 's/ *Name: *//')

case "$(basename $FC)" in
   amdflang*) OFFLOAD_FLAGS="--offload-arch=${ROCM_GPU}"; FREE_FORM_FLAG="-ffree-form" ;;
   flang*)    OFFLOAD_FLAGS="--offload-arch=${ROCM_GPU}"; FREE_FORM_FLAG="-Mfreeform" ;;
   gfortran*) OFFLOAD_FLAGS="--offload=-march=${ROCM_GPU}"; FREE_FORM_FLAG="-ffree-form" ;;
   *)         OFFLOAD_FLAGS=""; FREE_FORM_FLAG="-ffree-form" ;;
esac

TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

cat > "$TMPDIR/test.F90" <<'EOF'
program nowait_target_test
   use omp_lib
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none

   integer, parameter :: N     = 2**20
   integer, parameter :: K_GPU = 20000
   real(real64), allocatable :: a(:), b(:), c_sync(:), c_async(:), cpu_out(:)
   real(real64) :: t0, tA, tB, t_kernel, t_target_return
   real(real64) :: v
   integer :: i, k
   logical :: passed

   allocate(a(N), b(N), c_sync(N), c_async(N), cpu_out(N))
   do i = 1, N
      a(i) = real(i, real64) * 1.0e-6_real64
      b(i) = 2.0_real64
   end do

   ! warm-up
   !$omp target teams distribute parallel do map(to: a) map(from: c_sync)
   do i = 1, N
      c_sync(i) = a(i)
   end do
   !$omp end target teams distribute parallel do

   ! time the kernel synchronously
   t0 = omp_get_wtime()
   !$omp target teams distribute parallel do private(v, k) &
   !$omp&  map(to: a, b) map(from: c_sync)
   do i = 1, N
      v = a(i)
      do k = 1, K_GPU
         v = sin(v) + cos(v)
      end do
      c_sync(i) = v * b(i)
   end do
   !$omp end target teams distribute parallel do
   t_kernel = omp_get_wtime() - t0

   ! same kernel inside parallel + masked, with nowait
   t_target_return = 0.0_real64
   !$omp parallel private(i, v, k)
      !$omp masked
         tA = omp_get_wtime()
         !$omp target teams distribute parallel do nowait private(v, k) &
         !$omp&  map(to: a, b) map(from: c_async)
         do i = 1, N
            v = a(i)
            do k = 1, K_GPU
               v = sin(v) + cos(v)
            end do
            c_async(i) = v * b(i)
         end do
         !$omp end target teams distribute parallel do
         tB = omp_get_wtime()
         t_target_return = tB - tA
      !$omp end masked

      !$omp do schedule(dynamic, 1024)
      do i = 1, N
         v = a(i)
         do k = 1, K_GPU/100
            v = sin(v) + cos(v)
         end do
         cpu_out(i) = v
      end do
      !$omp end do
   !$omp end parallel

   passed = .true.
   do i = 1, N
      if (abs(c_sync(i) - c_async(i)) > 1.0e-12_real64) then
         passed = .false.
         exit
      end if
   end do

   print '(a, f10.4, a)', "kernel time alone            : ", t_kernel,        " s"
   print '(a, f10.4, a)', "masked thread held at nowait : ", t_target_return, " s"

   if (passed .and. t_target_return < 0.5_real64 * t_kernel) then
      print *, "PASS!"
   else if (.not. passed) then
      print *, "FAIL! (kernel results differ between sync and nowait)"
   else
      print '(a, f10.4, a, f10.4, a)', " FAIL! (masked thread held ", &
            t_target_return, " s on a ", t_kernel, " s kernel)"
   end if

   deallocate(a, b, c_sync, c_async, cpu_out)
end program nowait_target_test
EOF

$FC -O3 -fopenmp $FREE_FORM_FLAG $OFFLOAD_FLAGS "$TMPDIR/test.F90" -o "$TMPDIR/test"
OMP_NUM_THREADS=8 "$TMPDIR/test"
