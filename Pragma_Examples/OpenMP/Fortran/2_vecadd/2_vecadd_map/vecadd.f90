! Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
! 
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
! 
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
! 
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
program main

    use omp_lib 
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    ! Size of vectors
    integer :: n = 10000000
 
    ! Input vectors and Output vector
    real(real64),dimension(:),allocatable :: a, b, c
 
    integer :: i
    real(real64) :: sum
    real(real64) :: startt, endt


 
    ! Allocate memory for each vector
    allocate(a(n), b(n), c(n))
 
    ! Initialize input vectors.
    !$omp target teams distribute parallel do simd map(tofrom:a,b)
    do i=1,n
        a(i) = sin(dble(i)*1.0d0)*sin(dble(i)*1.0d0)
        b(i) = cos(dble(i)*1.0d0)*cos(dble(i)*1.0d0) 
        c(i) = 0.0d0
    enddo
    !$omp end target teams distribute parallel do simd

    !meassure after warmup kernel
    startt=omp_get_wtime()
    ! Sum each component of arrays

    !$omp target teams distribute parallel do simd map(to: a(1:n),b(1:n)) map(from: c(1:n))
    do i=1,n
        c(i) = a(i) + b(i)
    enddo
    !$omp end target teams distribute parallel do simd

    ! Sum up vector c. Print result divided by n. It should equal 1
    sum = 0.0d0
    !$omp target teams distribute parallel do simd map(to:c) reduction(+:sum)
    do i=1,n
        sum = sum +  c(i)
    enddo
    !$omp end target teams distribute parallel do simd

    sum = sum/dble(n)
    write(*,'("Final result: ",f10.6)') sum

    endt=omp_get_wtime()
    write(*,'("Runtime is: ",f8.6," secs")') endt-startt
 
    ! Deallocate memory
    deallocate(a, b, c)
 
end program
