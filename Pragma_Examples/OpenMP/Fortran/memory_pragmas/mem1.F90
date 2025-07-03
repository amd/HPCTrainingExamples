! Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:

! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.

! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
program main
  use omp_lib
  implicit none
  integer, parameter :: NTIMERS = 1
  integer :: num_iteration, n, i, iter
  real(8) :: main_timer, main_start, start
  real(8) :: sum_time, max_time, min_time, avg_time
  real(8), allocatable :: timers(:)
  real(8), allocatable :: x(:), y(:), z(:)
  real(8) :: a

  num_iteration = NTIMERS
  n = 100000
  main_start = omp_get_wtime()
  a = 3.0d0

  allocate(x(n), y(n), z(n), timers(num_iteration))

  do i = 1, n
    x(i) = 2.0d0
    y(i) = 1.0d0
  end do

  do iter = 1, num_iteration
    start = omp_get_wtime()
    call daxpy(n, a, x, y, z)
    timers(iter) = omp_get_wtime() - start
  end do

  max_time = -1.0d10
  min_time =  1.0d10
  sum_time =  0.0d0

  sum_time = sum(timers)
  max_time = maxval(timers)
  min_time = minval(timers)
  avg_time = sum_time / dble(num_iteration)

  print '(A,F9.6,A,F9.6,A,F9.6)', "-Timing in Seconds: min=", min_time, ", max=", max_time, ", avg=", avg_time
  main_timer = omp_get_wtime() - main_start
  print '(A,F9.5)', "-Overall time is ", main_timer
  print "(A, I0, A, F0.5)", "Last Value: z(", n, ")=", z(n)

  deallocate(x, y, z, timers)

contains

  subroutine daxpy(n, a, x, y, z)
    implicit none
    integer, intent(in) :: n
    real(8), intent(in) :: a
    real(8), intent(in) :: x(n), y(n)
    real(8), intent(out) :: z(n)
    integer :: i
    !$omp target teams distribute parallel do map(to: x, y) map(from: z)
    do i = 1, n
      z(i) = a * x(i) + y(i)
    end do
  end subroutine daxpy

end program main
