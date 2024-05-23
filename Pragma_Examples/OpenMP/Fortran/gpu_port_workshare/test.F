module overload_assignment_m
    implicit none
    private
    public assignment (=)
    public matrix

    integer,parameter :: rk=8
 
    type matrix
            real(kind=rk), dimension(:,:,:),allocatable :: val
    end type

    interface assignment (=)
        module procedure matrix_to_scalar
    end interface
contains
    subroutine matrix_to_scalar(mat, scal)

        type(matrix), intent (inout) :: mat
        real(kind=rk), intent (in)  :: scal
        integer :: i,j,k
        integer ::imin,imax,jmin,jmax,kmin,kmax
        real(kind=rk),dimension(:,:,:),allocatable :: tmp
        

        imin=lbound(mat%val,1); imax=ubound(mat%val,1)
        jmin=lbound(mat%val,2); jmax=ubound(mat%val,2)
        kmin=lbound(mat%val,3); kmax=ubound(mat%val,3)
        allocate(tmp(imin:imax,jmin:jmax,kmin:kmax))
        !$omp target teams distribute parallel do map(tofrom:tmp) map(to:scal)
        do k=kmin,kmax
        do j=jmin,jmax
        do i=imin,imax
          tmp(i,j,k) = scal
        end do
        end do
        end do
        !$omp end target teams distribute parallel do
        mat%val = tmp
        deallocate(tmp)
    end subroutine
end module

program main
    use overload_assignment_m
    implicit none

    integer,parameter :: rk=8
    type(matrix) :: a

    allocate(a%val(1:100,1:3,1:2))
    a = 1.0_rk

    print *, "a:", a%val
    deallocate(a%val)
end program
