! Original Author: Arjen Tamerus at DiRAC Hackathon Feb 2024
! Released to AMD as sample code
program gemm
        use gemm_mod

        implicit none

#ifdef LOCAL_ALLOC
        call do_gemm()
#else
        complex(kind=c_double_complex), dimension(:,:), allocatable :: A, B, C

        allocate(A(1024,1024))
        allocate(B(1024,1024))
        allocate(C(1024,1024))

   #ifndef SINGLE_DIRECTIVE
        !$omp target data map(to:a,b) map(from:c)
        !$omp target data use_device_addr(a,b,c)
   #endif
        call do_gemm(A, B, C)
   #ifndef SINGLE_DIRECTIVE
        !$omp end target data
        !$omp end target data
   #endif

        deallocate(A,B,C)
#endif

        print *,"Success"

end program gemm
