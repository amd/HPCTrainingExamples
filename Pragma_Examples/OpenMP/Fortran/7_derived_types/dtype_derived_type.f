! Original Author: Arjen Tamerus at DiRAC Hackathon Feb 2025
! Released to AMD as sample code
module dtype
        implicit none 
        type :: my_dtype
                integer :: s, e
                integer,dimension(:),allocatable :: values
        end type

end module

program offload_types
        use dtype
        implicit none
        ! $omp declare mapper (my_dtype :: v) map(v, v%values(s:e))

        type(my_dtype),target :: my_instance
        integer,dimension(:),pointer :: values_ptr
        integer :: i

        allocate(my_instance%values(1024))
        my_instance%s=1
        my_instance%e=1024


        !$omp target teams distribute parallel do  &
        !$omp              map(tofrom:my_instance)
        do i = 1,1024
                my_instance%values(i) = i
        end do


        write(*,*) my_instance%values(10)

        !$omp target exit data map(release:my_instance)

        deallocate(my_instance%values)

end program
