module norm_mod
  use kind_mod
  use mesh_mod, only: mesh_t
  implicit none

  private
  public :: norm

contains

  function norm(mesh, u) result(norm_val)
    type(mesh_t), intent(inout) :: mesh
    real(RK), intent(inout) :: u(:,:)
    real(RK) :: norm_val
    integer(IK) :: i,j,n_x,n_y
    real(RK) :: dxdy

    dxdy = mesh%dx*mesh%dy

    norm_val = 0._RK

    do j = 1,mesh%n_y
      do i = 1,mesh%n_x
        norm_val = norm_val + u(i,j)**2*dxdy
      end do
    end do

    norm_val = sqrt(norm_val)/(mesh%n_x*mesh%n_y)
  end function norm

end module norm_mod
