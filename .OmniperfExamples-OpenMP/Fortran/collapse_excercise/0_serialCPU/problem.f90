PROGRAM problem
  !use mpi_f08
  use omp_lib
  implicit none



  integer, parameter :: NI=200, NJ=200, NK=200, rk=8, maxrepeat=100
  integer :: i,j,k, i_rb, repeat
  real(kind=rk), allocatable, dimension(:,:,:) :: f
  real(kind=rk) :: starttime

  allocate(f(0:NI+1, 0:NJ+1, 0:NK+1))

  !initialisation
  do i_rb =1,2
    do k = 0,NK+1
      do j = 0,NJ+1
        do i = 1+MOD(j + k + i_rb + 1, 2), NI, 2
          f(i,j,k) = real(i_rb, kind=rk) 
        end do
      end do
    end do
  end do
 
  write(*,*) "initialization:" 
  write(*,*) f(1,1,1), f(2,1,1),f(3,1,1),f(4,1,1),"..."
  write(*,*) f(1,2,1), f(2,2,1),f(3,2,1),f(4,2,1),"..."
  write(*,*) f(1,3,1), f(2,3,1),f(3,3,1),f(4,3,1),"..."

  !timer

  starttime = omp_get_wtime() !MPI_WTime()
  !warm up loop
  do i_rb =1,2
    do k = 1,NK
      do j = 1,NJ
        do i = 1+MOD(j + k + i_rb, 2), NI, 2
          !this is a silly prototype for smoothing in a 7 point stencil
          f(i,j,k) = (f(i-1,j,k) + f(i+1,j,k) +      &
                      f(i,j-1,k) + f(i,j+1,k) +      &
                      f(i,j,k-1) + f(i,j,k+1) +      &
                      f(i,j,k)) / 7.0_rk
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
      do k = 1,NK
        do j = 1,NJ
          do i = 1+MOD(j + k + i_rb, 2), NI, 2
            !this is a silly prototype for smoothing in a 7 point stencil
            f(i,j,k) = (f(i-1,j,k) + f(i+1,j,k) +      &
                        f(i,j-1,k) + f(i,j+1,k) +      &
                        f(i,j,k-1) + f(i,j,k+1) +      &
                        f(i,j,k)) / 7.0_rk
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
