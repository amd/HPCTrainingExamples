! Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

program main

    use omp_lib 

    USE ISO_C_BINDING,   ONLY: c_null_char  
    
    implicit none


    !$omp requires unified_shared_memory

    ! Size of vectors
    integer,parameter :: n = 10000000
    integer,parameter :: Niter = 10
    ! Input vectors and Output vector
    real(8),dimension(:),allocatable :: a, b, c
    integer :: i,iter
    real(8) :: sum
    real(8) :: startt, endt


    startt=omp_get_wtime()

    ! Allocate memory for each vector
    allocate(a(n), b(n), c(n))
    
    DO iter = 1,Niter

      ! Initialize input vectors.
      !$omp target teams distribute parallel do simd
      do i=1,n
          a(i) = sin(real(i,kind=8)*1.0d0)*sin(real(i,kind=8)*1.0d0)
          b(i) = cos(real(i,kind=8)*1.0d0)*cos(real(i,kind=8)*1.0d0) 
          c(i) = 0.0d0
      enddo
      !$omp end target teams distribute parallel do simd

      !$omp target teams distribute parallel do simd
      do i=1,n
          c(i) = a(i) + b(i)
      enddo
      !$omp end target teams distribute parallel do simd
      
      ! Sum up vector c. Print result divided by n. It should equal 1
      sum = 0.0d0
      !$omp target teams distribute parallel do simd reduction(+:sum)
      do i=1,n
          sum = sum +  c(i)
      enddo
      !$omp end target teams distribute parallel do simd
  
      sum = sum/real(n,kind=8)

    END DO
    
    deallocate(a,b,c)
    write(*,'("Final result: ",f10.6)') sum

    endt=omp_get_wtime()
    write(*,'("Runtime is: ",f8.6," secs")') endt-startt
 
 
end program
