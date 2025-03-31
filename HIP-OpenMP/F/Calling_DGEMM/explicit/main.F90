! Copyright AMD 2024-2025, MIT License, contact Bob.Robey@amd.com
program example
    use rocm_interface
    use iso_c_binding
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    integer :: i, j
    real(real64) :: sum_check
    ! this corresponds to a no transpose in rocblas (112 is transpose)
    integer :: modea=111, modeb=111

    integer :: N=100
    real(real64) :: alpha=1.0, beta=0.0
    integer :: lda=100, ldb=100, ldc=100

    real(real64),allocatable,target,dimension(:,:) :: A, B, C
    type(c_ptr)                               :: rocblas_handle    !...

    allocate(A(N,N),B(N,N),C(N,N))
    call RANDOM_NUMBER(A)    ! Initialize matrices
    call RANDOM_NUMBER(C)    ! Initialize matrices
    B = 0
   
    ! init b to the identity matrix
    !$OMP target teams distribute parallel do 
    do i=1,N
       B(i,i) = 1
    end do   

    call init_rocblas(rocblas_handle)     ! Initialize rocBLAS

    !$OMP target enter data map(to:A,B,C)
    !$OMP target data use_device_addr(C,B,C)
    call omp_dgemm(rocblas_handle,modea,modeb,N,N,N,alpha,&
        c_loc(A),lda,c_loc(B),ldb,beta,c_loc(C),ldc)
    !$OMP end target data
    !$OMP target update from(C)
    !$OMP target exit data map(delete:A,B,C)

    
    sum_check = 0.0
    ! init b to the identity matrix
    !$OMP target teams distribute parallel do reduction(+:sum_check) collapse(2)
    do i=1,N
       do j=1,N
          sum_check = sum_check + abs(A(i,j) - C(i,j))
       end do
    end do   

    if (abs(sum_check) < 1.e-16) then
       print*, "PASSED!"
    else
       print* ,"FAILED!"
    endif     

end program example
