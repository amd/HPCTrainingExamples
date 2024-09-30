! This example was created by Johanna Potyka
! Copyright (c) 2024 AMD HPC Application Performance Team
! MIT License

      program device_routine
      !----------------------------
      ! description: this program is meant to demonstrate 
      !              how to call a device subroutine
         use omp_lib

         implicit none

         !---variables
         integer,parameter :: N=1000
         !N                   number of values in x array
         integer,parameter :: rk=kind(1.0d0)
         !rk                  kind of real
         integer :: k, err_stat
         !k         index
         !err_stat  status variable
         real(kind=rk),dimension(:),ALLOCATABLE :: x
         !x                                        array
         real(kind=rk) :: sum
         !sum             used to sum up x
         interface

                 subroutine compute(x)
                   !$omp declare target link(compute)
                   integer,parameter :: rk=8
                   real(kind=rk), intent(inout) :: x
                 end subroutine compute
         end interface



         !---allocation
         allocate(x(1:N),STAT=err_stat)
         if(err_stat /= 0) then
             write(*,*) "error while allocating"
             STOP
         end if
         
         !---initialisation
         x = -1.0_rk
         !--- call a device subroutine in kernel
!$omp target teams distribute parallel do simd map(tofrom:x)
         do k=1,N
            call compute(x(k))
            !x(k) = 1.0_rk
         end do
!$omp end target teams distribute parallel do simd         

         !--- initialize sum
        sum = 0.0_rk;

        !--- sum up x to sum on device with reduction
!$omp target teams distribute parallel do simd reduction(+:sum) map(to:x)
        do k=1,N
           sum = sum + x(k)
        end do
!$omp end target teams distribute parallel do simd        

        !--- print result
        Write(*,'(A,F0.12)') "Result: sum of x is ",sum

        call compute(x(1))
       deallocate(x) 

      end program device_routine
      
