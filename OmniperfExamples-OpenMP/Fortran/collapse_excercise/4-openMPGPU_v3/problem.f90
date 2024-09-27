PROGRAM problem

  use omp_lib
  implicit none

  !$omp requires unified_shared_memory

  integer, parameter :: NI=1024, NJ=1024, NK=1024, rk=8, maxrepeat=100
  integer :: i,j,k, i_rb, repeat
  real(kind=rk), allocatable, dimension(:,:,:) :: f
  real(kind=rk) :: starttime

  allocate(f(0:NI+1, 0:NJ+1, 0:NK+1))

  !initialisation
  do i_rb =1,2
    !!$omp target teams distribute parallel do collapse(3) private(i,j,k) shared(f,i_rb)
    !$omp target teams distribute parallel do collapse(3) private(i,j,k) shared(f,i_rb)
    do k = 0,NK+1
      do j = 0,NJ+1
        do i = 0,NI/2-1
          if(mod(i+j+k+i_rb,2) == 0) then
            f(i,j,k) = real(i_rb, kind=rk) 
          end if
        end do
      end do
    end do
  end do
 
  write(*,*) "initialization:" 
  write(*,*) f(1,1,1), f(2,1,1),f(3,1,1),f(4,1,1),"..."
  write(*,*) f(1,2,1), f(2,2,1),f(3,2,1),f(4,2,1),"..."
  write(*,*) f(1,3,1), f(2,3,1),f(3,3,1),f(4,3,1),"..."

  !timer

  starttime = omp_get_wtime()
  !warm up loop
  do i_rb =1,2
    !!$omp target teams distribute parallel do collapse(3) private(i,j,k) shared(f,i_rb)
    !$omp target teams distribute parallel do collapse(3) private(i,j,k) shared(f,i_rb)
    do k = 1,NK
      do j = 1,NJ
        do i = 1,NI
          if(mod(i+j+k+i_rb,2) == 0) then
            !this is a silly prototype for smoothing in a 7 point stencil
            f(i,j,k) = (f(i-1,j,k) + f(i+1,j,k) +      &
                         f(i,j-1,k) + f(i,j+1,k) +      &
                         f(i,j,k-1) + f(i,j,k+1) +      &
                         f(i,j,k)) / 7.0_rk
          end if
        end do
      end do
    end do
  end do
  write(*,*) "warmup loop time:",(omp_get_wtime()-starttime)*1000.0_rk,"ms"
  !reset timer:
  
  starttime=omp_get_wtime()
  !time 100 iterations  after warmup loop:
  do repeat=1,maxrepeat-1
    do i_rb =1,2
      !!$omp target teams distribute parallel do collapse(3) private(i,j,k) shared(f,i_rb)
      !$omp target teams distribute parallel do collapse(3) private(i,j,k) shared(f,i_rb)
      do k = 1,NK
        do j = 1,NJ
          do i = 1,NI
            !ii = 2*i + mod(i+j+k+i_rb,2)
            if(mod(i+j+k+i_rb,2) == 0) then
              !this is a silly prototype for smoothing in a 7 point stencil
              f(i,j,k) = (f(i-1,j,k) + f(i+1,j,k) +      &
                           f(i,j-1,k) + f(i,j+1,k) +      &
                           f(i,j,k-1) + f(i,j,k+1) +      &
                           f(i,j,k)) / 7.0_rk
           end if
          end do
        end do
      end do
    end do
  end do
  

  write(*,*) "after",maxrepeat,"red and black smoothing steps:"
  write(*,*) f(1,1,1), f(2,1,1),f(3,1,1),f(4,1,1),"..."
  write(*,*) f(1,2,1), f(2,2,1),f(3,2,1),f(4,2,1),"..."
  write(*,*) f(1,3,1), f(2,3,1),f(3,3,1),f(4,3,1),"..."
  
  write(*,*) "time", (omp_get_wtime()-starttime)/maxrepeat*1000.0_rk,"ms"
  
  deallocate(f)

END PROGRAM        
