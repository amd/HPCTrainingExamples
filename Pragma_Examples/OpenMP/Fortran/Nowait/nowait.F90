! Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! Test that the OpenMP `nowait` clause on a
! `target teams distribute parallel do` construct actually allows the
! encountering thread to continue past the kernel launch BEFORE the GPU
! kernel completes, so that the host can do other work while the GPU is
! still running.
!
! Pattern (from the OpenMP 6.0 spec, "nowait clause" example):
!
!    !$omp parallel
!       !$omp masked
!          !$omp target teams distribute parallel do nowait &
!          !$omp&    map(to:...) map(from:...)
!          do i = ...
!             ! GPU work
!          end do
!          !$omp end target teams distribute parallel do
!       !$omp end masked
!
!       !$omp do schedule(dynamic, chunk)
!       do i = ...
!          ! CPU work, masked thread joins here after the async launch
!       end do
!       !$omp end do
!    !$omp end parallel
!
! Strategy
! --------
! 1. Time the same GPU kernel run synchronously (no nowait) to obtain
!    t_kernel.
! 2. Run the spec-style pattern above and measure how long the masked
!    thread is held at the `target ... nowait` line, t_target_return.
! 3. PASS iff the GPU kernel produced the same output as the sync run
!    AND t_target_return is much smaller than t_kernel (the masked
!    thread came back well before the kernel completed).
!
program nowait_target_test
   use omp_lib
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none

   integer, parameter :: N     = 2**20
   integer, parameter :: K_GPU = 20000
   real(real64), allocatable :: a(:), b(:), c_sync(:), c_async(:), cpu_out(:)
   real(real64) :: t0, tA, tB, t_kernel, t_target_return, t_total
   real(real64) :: v
   integer :: i, k
   logical :: passed

   allocate(a(N), b(N), c_sync(N), c_async(N), cpu_out(N))
   do i = 1, N
      a(i) = real(i, real64) * 1.0e-6_real64
      b(i) = 2.0_real64
   end do

   ! Warm-up to absorb device-init cost
   !$omp target teams distribute parallel do map(to: a) map(from: c_sync)
   do i = 1, N
      c_sync(i) = a(i)
   end do
   !$omp end target teams distribute parallel do

   ! ---- 1. Calibrate: time the kernel alone (synchronous) ----
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

   ! ---- 2. Spec-style pattern: parallel + masked + target nowait ----
   t_target_return = 0.0_real64
   t0 = omp_get_wtime()
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
         ! If `nowait` is honored, tB-tA is microseconds, not the full
         ! kernel time.
         t_target_return = tB - tA
      !$omp end masked

      ! All threads (including the masked one, after it falls through)
      ! participate in this CPU loop while the GPU kernel is in flight.
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
   ! implicit barrier above also waits for the deferred target task
   t_total = omp_get_wtime() - t0

   ! ---- 3. Correctness: same kernel, same inputs => identical output ----
   passed = .true.
   do i = 1, N
      if (abs(c_sync(i) - c_async(i)) > 1.0e-12_real64) then
         passed = .false.
         exit
      end if
   end do

   print '(a, f10.4, a)', "Calibrated kernel time alone         : ", t_kernel,         " s"
   print '(a, f10.4, a)', "masked thread held at target nowait  : ", t_target_return,  " s"
   print '(a, f10.4, a)', "Total parallel region time           : ", t_total,          " s"

   if (passed .and. t_target_return < 0.5_real64 * t_kernel) then
      print *, "PASS!"
   else if (.not. passed) then
      print *, "FAIL! (kernel results differ between sync and nowait variants)"
   else
      print '(a, f10.4, a, f10.4, a)', " FAIL! (masked thread held ", &
            t_target_return, " s on a ", t_kernel, " s kernel; nowait not honored)"
   end if

   deallocate(a, b, c_sync, c_async, cpu_out)
end program nowait_target_test
