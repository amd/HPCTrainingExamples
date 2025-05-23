! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

program main

    use omp_lib

    use iso_fortran_env, only: real64

    implicit none

    !$omp requires unified_shared_memory

    ! Size of vectors
    integer,parameter :: n = 10000000
    integer,parameter :: Niter = 10
    ! Input vectors and Output vector
    real(kind=real64),dimension(:),allocatable :: a, b, c
    integer :: i,iter
    real(kind=real64) :: sum
    real(kind=real64) :: startt, endt

    startt=omp_get_wtime()

    ! Allocate memory for each vector
    allocate(a(n), b(n), c(n))
    
    do iter = 1,Niter

      ! Initialize input vectors.
      !$omp target teams distribute parallel do simd
      do i=1,n
          a(i) = sin(dble(i))*sin(dble(i))
          b(i) = cos(dble(i))*cos(dble(i))
          c(i) = 0.0_real64
      enddo

      !$omp target teams distribute parallel do simd
      do i=1,n
          c(i) = a(i) + b(i)
      enddo

      ! Sum up vector c. Print result divided by n. It should equal 1
      sum = 0.0_real64
      !$omp target teams distribute parallel do simd reduction(+:sum)
      do i=1,n
          sum = sum +  c(i)
      enddo

      sum = sum/dble(n)

    end do

    deallocate(a,b,c)
    write(*,'("Final result: ",f10.6)') sum

    endt=omp_get_wtime()
    write(*,'("Runtime is: ",f10.4," msecs")') (endt-startt)*1000.0_real64
end program
