! Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

    DO iter = 1,Niter
      ! Allocate memory for each vector
      allocate(a(n), b(n), c(n))
         
      ! Initialize input vectors.
      !$omp target teams distribute parallel do simd
      do i=1,n
          a(i) = sin(dble(i))*sin(dble(i))
          b(i) = cos(dble(i))*cos(dble(i)) 
          c(i) = 0.0d0
      enddo

      !$omp target teams distribute parallel do simd
      do i=1,n
          c(i) = a(i) + b(i)
      enddo
      
      ! Sum up vector c. Print result divided by n. It should equal 1
      sum = 0.0d0
      !$omp target teams distribute parallel do simd reduction(+:sum)
      do i=1,n
          sum = sum +  c(i)
      enddo
  
      sum = sum/dble(n)

      deallocate(a,b,c)
    END DO
    
    write(*,'("Final result: ",f10.6)') sum

    endt=omp_get_wtime()
    write(*,'("Runtime is: ",f8.6," secs")') endt-startt
 
end program
