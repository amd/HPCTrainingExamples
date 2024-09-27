! Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
! 
program main

    use omp_lib 
    implicit none
    !$omp requires unified_shared_memory
    ! Size of vectors
    integer :: n = 10000000
 
    ! Input vectors and Output vector
    real(8),dimension(:),allocatable :: a, b, c
 
    integer :: i
    real(8) :: sum
    real(8) :: startt, endt


 
    ! Allocate memory for each vector
    allocate(a(n), b(n), c(n))
 
    ! Initialize input vectors.
    !$omp target teams distribute parallel do simd nowait depend(out:a,b)
    do i=1,n
        a(i) = sin(dble(i)*1.0d0)*sin(dble(i)*1.0d0)
        b(i) = cos(dble(i)*1.0d0)*cos(dble(i)*1.0d0) 
    enddo
    !$omp end target teams distribute parallel do simd

    !meassure after warmup kernel
    startt=omp_get_wtime()
    ! Sum each component of arrays

    !$omp target teams distribute parallel do simd nowait depend(in:a,b) depend(out:c)
    do i=1,n
        c(i) = a(i) + b(i)
    enddo
    !$omp end target teams distribute parallel do simd

    ! Sum up vector c. Print result divided by n. It should equal 1
    sum = 0.0d0
    !$omp target teams distribute parallel do simd reduction(+:sum) nowait depend(in:c)
    do i=1,n
        sum = sum +  c(i)
    enddo
    !$omp end target teams distribute parallel do simd

    !$omp taskwait
    sum = sum/dble(n)
    write(*,'("Final result: ",f10.6)') sum

    endt=omp_get_wtime()
    write(*,'("Runtime is: ",f8.6," secs")') endt-startt
 
    ! Deallocate memory
    deallocate(a, b, c)
 
end program
