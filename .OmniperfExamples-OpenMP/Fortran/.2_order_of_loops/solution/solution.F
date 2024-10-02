        PROGRAM problem

        USE omp_lib

        IMPLICIT NONE

        !$OMP REQUIRES UNIFIED_SHARED_MEMORY

        INTEGER,PARAMETER :: n = 20480
        INTEGER,PARAMETER :: m = 20480

        INTEGER,PARAMETER :: rk = 8

        REAL(KIND=rk), DIMENSION(:), ALLOCATABLE   :: y,x
        REAL(KIND=rk), DIMENSION(:,:), ALLOCATABLE :: A
        REAL(KIND=rk) :: res, temp, starttime
        INTEGER :: i,j,status!,k

        status=0
        
        !Allocate
        ALLOCATE(y(1:n), &
                 x(1:m), &
                A(1:m,1:n),STAT=status)

         y = 1.0_rk
         x = 1.0_rk
         A = 1.0_rk

         res = 0.0_rk
         temp = 0.0_rk
         !warm up loop
         !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO DEFAULT(SHARED) &
         !$OMP PRIVATE(i,j,temp) REDUCTION(+:res)
         DO i=1,n
           temp = 0.0_rk
           DO j=1,m
             temp = temp + A(j,i) * x(j)
           END DO
           res = res + y(i) * temp
         END DO
         !$OMP END TARGET TEAMS DISTRIBUTE PARALLEL DO
         
         res = 0.0_rk

         !timinig loop
         starttime = omp_get_wtime()
         !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO DEFAULT(SHARED) &
         !$OMP PRIVATE(i,j,temp) REDUCTION(+:res)
         DO i=1,n
           temp = 0.0_rk
           DO j=1,m
             temp = temp + A(j,i) * x(j)
           END DO
            res = res + y(i) * temp
         END DO

         WRITE(*,*)  "yAx time",(omp_get_wtime()-starttime)*1.0e3_rk,"ms"

         IF(ABS(res - REAL(n,KIND=rk)*REAL(m, KIND=rk)) > 0.0001) THEN
                 WRITE(*,*) "the result is incorrect:"
                 WRITE(*,*) "result", res,"expected result", REAL(n,KIND=rk)*REAL(m, KIND=rk)
         END IF


        END PROGRAM problem
