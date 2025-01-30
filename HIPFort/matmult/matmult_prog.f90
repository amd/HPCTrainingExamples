program matmult
        use matmult_mod
        use matmult_interface

        implicit none

        integer(c_int)     :: n
        real(kind=c_double), dimension(:,:), allocatable :: A, B, C1, C2

        ! with this example, we are computing C=A*B
        ! using hipblas and also doing it with a simple hip kernel
        ! where each thread computes one entry of C
        ! then we compare the results and show that they match

        n = 1024
        allocate(A(n,n))
        allocate(B(n,n))
        allocate(C1(n,n))
        allocate(C2(n,n))

        ! initialize matrices
        call RANDOM_NUMBER(A)
        call RANDOM_NUMBER(B)
        C1 = 0.0;
        C2 = 0.0;

        ! here we use hipfort to call a function from hipblas
        ! the arrays have to be on device to be fed to hipblas
        ! hence we use openmp to move them on the device
        !$OMP target enter data map(to:a,b,c1)
        !$OMP target data use_device_ptr(a,b,c1)
        call do_matmult(c_loc(A), c_loc(B), c_loc(C1), n)
        !$OMP end target data
        !$OMP target update from(c1)

        ! here we are calling the HIP kernel leveraging the "matmult_interface" 
        call matrix_multiply(c_loc(A), c_loc(B), c_loc(C2), n)

        !$OMP target exit data map(delete:a,b,c1)

        ! here we compute the absolute value of the
        ! max entry of C2-C1 
        if (MAXVAL(ABS(C2-C1)) < 1.e-12) then
           print*, "Success"
        else 
           print*, "Failure"      
        endif

end program matmult
