! This program is based on a reproducer created by Mahdieh Ghazimirsaeed
! Copyright (c) 2025 AMD HPC Application Performance Team
! MIT License
      program reduction
      implicit none

      integer:: i,j,k
      integer, parameter :: real8 = kind(0.0d0)
      real(real8):: ce1,ce2

      ce1=0.0d0
      ce2=0.0d0
      do j=1, 1000
         ce1=ce1+1
         ce2=ce2+1
      enddo

      write(*,'(a," ",f14.8," ",a,f14.8)') "ce1 = ", ce1, "ce2 = ", ce2
      end program
