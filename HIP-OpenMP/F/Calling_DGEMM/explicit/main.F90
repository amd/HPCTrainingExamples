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

    real(real64),allocatable,target,dimension(:,:) :: a, b, c
    type(c_ptr)                               :: rocblas_handle    !...

    allocate(a(N,N),b(N,N),c(N,N))
    call RANDOM_NUMBER(a)    ! Initialize matrices
    call RANDOM_NUMBER(c)    ! Initialize matrices
    b = 0
   
    ! init b to the identity matrix
    !$OMP target teams distribute parallel do 
    do i=1,N
       b(i,i) = 1
    end do   

    call init_rocblas(rocblas_handle)     ! Initialize rocBLAS

    !$OMP target enter data map(to:a,b,c)
    !$OMP target data use_device_addr(a,b,c)
    call omp_dgemm(rocblas_handle,modea,modeb,N,N,N,alpha,&
        c_loc(a),lda,c_loc(b),ldb,beta,c_loc(c),ldc)
    !$OMP end target data
    !$OMP target update from(c)
    !$OMP target exit data map(delete:a,b,c)

    
    sum_check = 0.0
    ! init b to the identity matrix
    !$OMP target teams distribute parallel do reduction(+:sum_check) collapse(2)
    do i=1,N
       do j=1,N
          sum_check = sum_check + abs(a(i,j) - c(i,j))
       end do
    end do   

    if (abs(sum_check) < 1.e-16) then
       print*, "PASSED!"
    else
       print* ,"FAILED!"
    endif     

end program example
