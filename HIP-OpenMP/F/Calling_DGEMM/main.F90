! Copyright AMD 2024-2025, MIT License, contact Bob.Robey@amd.com
program example
    use rocm_interface
    use iso_c_binding
    use, intrinsic :: iso_fortran_env, only: real64
    implicit none
    integer :: modea=1, modeb=1
    integer :: M=100, N=100, K=100
    real(real64) :: alpha=1.0, beta=1.0
    integer :: lda=16,ldb=16, ldc=16
    real(real64),allocatable,target,dimension(:,:) :: a, b, c
    type(c_ptr)                               :: rocblas_handle    !...

    allocate(a(M,N),b(N,K),c(M,K))
    call RANDOM_NUMBER(a)    ! Initialize matrices
    call RANDOM_NUMBER(b)    ! Initialize matrices
    call RANDOM_NUMBER(c)    ! Initialize matrices
    call init_rocblas(rocblas_handle)     ! Initialize rocBLAS
    !...

    !$OMP target enter data map(to:a,b,c)
    !$OMP target data use_device_addr(a,b,c)
    call omp_dgemm(rocblas_handle,modea,modeb,M,N,K,alpha,&
        c_loc(a),lda,c_loc(b),ldb,beta,c_loc(c),ldc)
    !$OMP end target data
    !$OMP target update from(c)
    !$OMP target exit data map(delete:a,b,c)
    !...
end program example
