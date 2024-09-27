! Original Author: Arjen Tamerus at DiRAC Hackathon Feb 2024
! Released to AMD as sample code
module dtype

        type :: my_dtype
                integer :: s, e
                integer,dimension(:),allocatable :: values
        end type

end module

program offload_types
        use dtype

        type(my_dtype),target :: my_instance
        integer,dimension(:),pointer :: values_ptr
        integer :: i

        allocate(my_instance%values(1024))
        my_instance%s=1
        my_instance%e=1024

        values_ptr => my_instance%values

        !$omp target enter data map(to:my_instance,my_instance%values)

        !$omp target teams distribute parallel do shared(values_ptr)
        do i = 1,1024
                values_ptr(i) = i
        end do

        !$omp target exit data map(from:my_instance%values)

        write(*,*) my_instance%values(10)

        !$omp target exit data map(release:my_instance)

        deallocate(my_instance%values)

end program
