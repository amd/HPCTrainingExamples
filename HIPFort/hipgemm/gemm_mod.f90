! Original Author: Arjen Tamerus at DiRAC Hackathon Feb 2024
! Released to AMD as sample code
module gemm_mod
        use hipfort
        use hipfort_hipblas
        use iso_c_binding
        implicit none

        public :: do_gemm

        contains

#ifdef LOCAL_ALLOC
        subroutine do_gemm()
                complex(kind=c_double_complex), dimension(:,:), allocatable :: A, B, C
#else
        subroutine do_gemm(A, B, C)
                complex(kind=c_double_complex), dimension(:,:),intent(inout) :: A, B, C
#endif

                integer(kind=c_int) :: m, n, k, lda, ldb, ldc
                complex(kind=c_double_complex) :: alpha, beta

                type(c_ptr) :: handle

                integer(kind(hipblas_op_n)) :: ta, tb

                integer :: status

                ta = HIPBLAS_OP_N
                tb = HIPBLAS_OP_N

                m = 1024
                n = 1024
                k = 1024
                lda = 1024
                ldb = 1024
                ldc = 1024

#ifdef LOCAL_ALLOC
                allocate(A(m,k))
                allocate(B(k,n))
                allocate(C(m,n))
#endif

                alpha = 1.0
                beta = 1.0

                a = cmplx(1.0,0.0,c_double_complex)
                b = cmplx(2.0,0.0,c_double_complex)
                c = cmplx(0.0,0.0,c_double_complex)

                status = hipblasCreate(handle)
                write(0,*) "CREATE", status, status .eq. HIPBLAS_STATUS_SUCCESS

#ifdef SINGLE_DIRECTIVE
                !$omp target data map(to:a,b) map(from:c) use_device_ptr(a,b,c)
#else
                !$omp target data map(to:a,b) map(from:c)
                !$omp target data use_device_ptr(a,b,c)
#endif
                status = hipblasZgemm(handle, ta, tb, &
                        m, n, k, alpha, &
                        A, lda, B, ldb , beta, C, ldC)
#ifdef SINGLE_DIRECTIVE
                !$omp end target data
#endif

                status = hipdevicesynchronize()
                status = hipblasDestroy(handle)
#ifndef SINGLE_DIRECTIVE
                !$omp end target data
                !$omp end target data
#endif

                write(0,*) "ZGEMM", status, status .eq. HIPBLAS_STATUS_SUCCESS

                write(0,*) "C(1,1):", C(1,1)

        end subroutine

end module
