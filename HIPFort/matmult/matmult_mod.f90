! Original Author: Arjen Tamerus at DiRAC Hackathon Feb 2024
! Released to AMD as sample code
module matmult_mod
        use hipfort
        use hipfort_hipblas
        use iso_c_binding
        implicit none

        public :: do_matmult

        contains

        subroutine do_matmult(A, B, C, n)
                type(c_ptr),value :: A, B, C
                integer(kind=c_int), intent (in) :: n 
                integer(kind=c_int) :: lda, ldb, ldc
                real(kind=c_double) :: alpha, beta

                type(c_ptr) :: handle

                integer(kind(hipblas_op_n)) :: ta, tb

                integer :: status

                ta = HIPBLAS_OP_N
                tb = HIPBLAS_OP_N

                lda = n 
                ldb = n
                ldc = n

                alpha = 1.0
                beta = 0.0

                status = hipblasCreate(handle)

                status = hipblasDgemm(handle, ta, tb, &
                        n, n, n, alpha, &
                        A, lda, B, ldb , beta, C, ldC)

                status = hipdevicesynchronize()
                status = hipblasDestroy(handle)

        end subroutine

end module
