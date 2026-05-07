! Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
! This follows the pattern from the OpenMP 6.0 examples document
! regarding the nowait clause
! (https://www.openmp.org/wp-content/uploads/openmp-examples-6.0.pdf):
!
!    !$omp parallel
!    !$omp single
!       ! independent producers
!       !$omp target ... nowait depend(out: a) ...
!       do i = ...; a(i) = ...; end do
!
!       !$omp target ... nowait depend(out: b) ...
!       do i = ...; b(i) = ...; end do
!
!       ! consumer -- waits for both producers via depend(in: a, b)
!       !$omp target ... nowait depend(in: a, b) depend(out: c) ...
!       do i = ...; c(i) = a(i) + b(i); end do
!
!       ! host work -- runs concurrently with the GPU kernels
!       ...
!
!       !$omp taskwait
!    !$omp end single
!    !$omp end parallel
!
! This example is here to demonstrate that, with `nowait`, the host
! thread does not block at the end of a `target` construct: it returns
! immediately and is free to issue more kernels and run host code
! the GPU is still busy. 
!
program nowait_multi
   use omp_lib
   use, intrinsic :: iso_fortran_env, only: real64
   implicit none

   integer, parameter :: N = 2**20
   real(real64), allocatable :: a(:), b(:), c(:)
   real(real64) :: host_sum, t0, t_total
   real(real64) :: cmin, cmax, s
   integer :: i
   integer(kind=8) :: n_iter

   allocate(a(N), b(N), c(N))

   t0 = omp_get_wtime()

   !$omp parallel
   !$omp single
      ! Kernel 1: produce a(i) = sin(i)**2.
      ! depend(out: a) declares this task as a producer of a.
      !$omp target teams distribute parallel do nowait &
      !$omp&   depend(out: a) map(from: a)
      do i = 1, N
         a(i) = sin(real(i, real64))**2
      end do
      !$omp end target teams distribute parallel do

      ! Kernel 2: produce b(i) = cos(i)**2.
      ! No depend ordering with kernel 1, so the runtime is free to run
      ! them concurrently on the GPU if it can.
      !$omp target teams distribute parallel do nowait &
      !$omp&   depend(out: b) map(from: b)
      do i = 1, N
         b(i) = cos(real(i, real64))**2
      end do
      !$omp end target teams distribute parallel do

      ! Kernel 3: consume a and b, produce c.
      ! depend(in: a, b) makes this task wait for kernels 1 and 2.
      !$omp target teams distribute parallel do nowait &
      !$omp&   depend(in: a, b) depend(out: c) &
      !$omp&   map(to: a, b) map(from: c)
      do i = 1, N
         c(i) = a(i) + b(i)
      end do
      !$omp end target teams distribute parallel do

      ! Host work: independent of a, b, c, so the depend graph allows
      ! it to overlap with the GPU kernels.
      s = 0.0_real64
      do n_iter = 0_8, 5000000_8 - 1_8
         s = s + sin(real(n_iter, real64))
      end do
      host_sum = s

      ! Wait for all deferred target tasks to complete before reading
      ! c(:) outside the single block.
      !$omp taskwait
   !$omp end single
   !$omp end parallel

   t_total = omp_get_wtime() - t0

   cmin = c(1); cmax = c(1)
   do i = 1, N
      if (c(i) < cmin) cmin = c(i)
      if (c(i) > cmax) cmax = c(i)
   end do

   print '(a, f10.6)',   "c(1)    = ", c(1)
   print '(a, f10.6)',   "c(N)    = ", c(N)
   print '(a, f10.6)',   "min(c)  = ", cmin
   print '(a, f10.6)',   "max(c)  = ", cmax
   print '(a)',          "(every c(i) should be 1.0: sin^2 + cos^2 = 1)"
   print '(a, es12.4)',  "host_sum (concurrent CPU work)  = ", host_sum
   print '(a, f10.4, a)', "total elapsed time              = ", t_total, " s"

   deallocate(a, b, c)
end program nowait_multi
