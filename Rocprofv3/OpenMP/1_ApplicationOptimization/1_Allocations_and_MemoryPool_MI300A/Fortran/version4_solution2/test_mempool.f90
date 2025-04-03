! Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

module roctx

    implicit none

    private
    integer :: n = 0

    public :: roctxpush, roctxpop

 interface
     subroutine roctxrangepush(message) bind(c, name="roctxRangePushA")
       use iso_c_binding,   only: c_char
       implicit none
       character(c_char) :: message(*)
     end subroutine roctxrangepush

     subroutine roctxrangepop() bind(c, name="roctxRangePop")
       implicit none
     end subroutine roctxrangepop

 end interface

 contains

   subroutine roctxPush(name)
      character(len=*),intent(in) :: name
      n = n + 1
      call roctxRangePush(name)
   end subroutine roctxPush

   subroutine roctxPop(name)
      character(len=*),intent(in) :: name
      n = n - 1
      ! Print the marker name if there are more pop calls than push calls
      if (n < 0) then
          print *, "invalid pop for: ", name
          return
      endif
      call roctxRangePop()
   end subroutine roctxPop

end module



program main

    use omp_lib 

    use roctx
    
    !use Umpire Fortran API:
    USE umpire_mod
    
    USE ISO_C_BINDING,   ONLY: c_null_char  
    
    implicit none

    type(UmpireAllocator) base_allocator
    type(UmpireAllocator) mem_pool
    type(UmpireResourceManager) res_manager


    !$omp requires unified_shared_memory

    ! Size of vectors
    integer :: n = 10000000
    integer,parameter :: Niter = 10
    ! Input vectors and Output vector
    real(8),dimension(:),pointer,contiguous :: a, b, c
    integer :: i,iter
    real(8) :: sum
    real(8) :: startt, endt


    startt=omp_get_wtime()
    
    !Umpire pool definition
    res_manager = res_manager%get_instance()
    base_allocator = res_manager%get_allocator_by_name("HOST")
    !depending on the size of allocations it it is reccomended to either use a quick(small allocations) or a list (large allocations) pool.
    !mem_pool = res_manager%make_allocator_quick_pool("HOST_POOL", base_allocator, 512_8*1024_8, 1024_8)
    mem_pool = res_manager%make_allocator_list_pool("HOST_POOL", base_allocator, 3*10000000_8, 1024_8*1024_8)


    DO iter = 1,Niter
      call roctxpush("allocate" // c_null_char)      
      ! Allocate memory for each vector
      call mem_pool%allocate(a,[n])
      call mem_pool%allocate(b,[n])
      call mem_pool%allocate(c,[n])
      call roctxpop("allocate" // c_null_char)      
      
      call roctxpush("first touch" // c_null_char)      
      ! Initialize input vectors.
      !$omp target teams distribute parallel do simd
      do i=1,n
          a(i) = sin(real(i,kind=8)*1.0d0)*sin(real(i,kind=8)*1.0d0)
          b(i) = cos(real(i,kind=8)*1.0d0)*cos(real(i,kind=8)*1.0d0) 
          c(i) = 0.0d0
      enddo
      !$omp end target teams distribute parallel do simd
      call roctxpop("first touch" // c_null_char)      

      call roctxpush("element sum kernel" // c_null_char)      
      !$omp target teams distribute parallel do simd
      do i=1,n
          c(i) = a(i) + b(i)
      enddo
      !$omp end target teams distribute parallel do simd
      call roctxpop("element sum kernel" // c_null_char)      
      
      call roctxpush("global sum" // c_null_char)      
      ! Sum up vector c. Print result divided by n. It should equal 1
      sum = 0.0d0
      !$omp target teams distribute parallel do simd reduction(+:sum)
      do i=1,n
          sum = sum +  c(i)
      enddo
      !$omp end target teams distribute parallel do simd
  
      sum = sum/real(n,kind=8)
      call roctxpop("global sum" // c_null_char)      

      call roctxpush("deallocate" // c_null_char)      
      call mem_pool%deallocate(a)
      call mem_pool%deallocate(b)
      call mem_pool%deallocate(c)
      call roctxpop("deallocate" // c_null_char)      
    END DO
    
    write(*,'("Final result: ",f10.6)') sum

    endt=omp_get_wtime()
    write(*,'("Runtime is: ",f8.6," secs")') endt-startt
 
 
end program
