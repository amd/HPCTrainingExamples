program loop
        integer,dimension(1024) :: mydata
        integer  :: i

        !$omp target teams loop
        do i=1,1024
                mydata(i) = i
        end do
        !$omp end target teams loop

        print *,"Success"

end program loop
