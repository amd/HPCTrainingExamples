module matmult_interface
    interface
        subroutine matrix_multiply(A,B,C,n) bind(C)
            use iso_c_binding
            type(c_ptr),value  :: A,B,C
            integer(c_int)     :: n 
        end subroutine matrix_multiply 
    end interface
end module matmult_interface
