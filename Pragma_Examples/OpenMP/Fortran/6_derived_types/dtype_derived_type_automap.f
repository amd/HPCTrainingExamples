!Johanna Potyka 26th Sept 2025 added naive implementation for exercise for MI300A 

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


        type(my_dtype),target :: my_instance
        integer :: i

        allocate(my_instance%values(1024))
        my_instance%s=1
        my_instance%e=1024


        !$omp target teams distribute parallel do shared(my_instance)
        do i = my_instance%s,my_instance%e
                my_instance%values(i) = i
        end do

        write(*,*) my_instance%values(10)

        deallocate(my_instance%values)

end program
