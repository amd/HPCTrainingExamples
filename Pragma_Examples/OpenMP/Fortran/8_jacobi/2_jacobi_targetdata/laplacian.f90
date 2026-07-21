! Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
! This software is distributed under the MIT License
!
module laplacian_mod
  use kind_mod
  use mesh_mod, only: mesh_t
  implicit none

  private
  public :: laplacian, fused_laplacian_update

contains

  subroutine laplacian(mesh,u,au)
    type(mesh_t), intent(inout) :: mesh
    real(RK), intent(inout) :: u(:,:)
    real(RK), intent(inout) :: au(:,:)
    integer(IK) :: i,j
    real(RK) :: invdx2,invdy2

    invdx2 = mesh%dx**-2
    invdy2 = mesh%dy**-2

    !$omp target teams distribute parallel do collapse(2)
    do j = 2,mesh%n_y-1
      do i = 2,mesh%n_x-1
        au(i,j) = (-u(i-1,j)+2._RK*u(i,j)-u(i+1,j))*invdx2 &
                + (-u(i,j-1)+2._RK*u(i,j)-u(i,j+1))*invdy2
      end do
    end do

  end subroutine laplacian

  subroutine fused_laplacian_update(mesh,u,au,rhs,res)
    type(mesh_t), intent(inout) :: mesh
    real(RK), intent(inout) :: u(:,:), au(:,:), rhs(:,:), res(:,:)
    integer(IK) :: i,j,n_x,n_y,id
    real(RK) :: invdx2,invdy2, factor, temp
    real(RK) :: lap_val, rhs_bc

    n_x = mesh%n_x
    n_y = mesh%n_y
    invdx2 = mesh%dx**-2
    invdy2 = mesh%dy**-2
    factor = (2._RK*invdx2+2._RK*invdy2)**-1

    ! Fused compute: laplacian + update for interior in one kernel
    !$omp target teams distribute parallel do collapse(2)
    do j = 2, n_y-1
      do i = 2, n_x-1
        lap_val = (-u(i-1,j)+2._RK*u(i,j)-u(i+1,j))*invdx2 + &
                (-u(i,j-1)+2._RK*u(i,j)-u(i,j+1))*invdy2

        au(i,j) = lap_val
        temp = rhs(i,j) - lap_val
        res(i,j) = temp
        u(i,j) = u(i,j) + temp*factor
      end do
    end do

    ! Boundary conditions + update for boundary points - same logic as boundary_conditions.f90
    invdx2 = mesh%dx**-2
    invdy2 = mesh%dy**-2

    !$omp target teams distribute parallel do private(i,j,lap_val,temp)
    do id=1,2*n_x+2*n_y-4
      if (id == 1) then
        au(1,1) = (2._RK*u(1,1)-u(2,1))*invdx2 &
                + (2._RK*u(1,1)-u(1,2))*invdy2
        temp = rhs(1,1) - au(1,1)
        res(1,1) = temp
        u(1,1) = u(1,1) + temp*factor
      else if (id <= n_x-1) then
        i = id
        au(i,1) = (-u(i-1,1)+2._RK*u(i,1)-u(i+1,1))*invdx2 &
                + (2._RK*u(i,1)-u(i,2))*invdy2
        temp = rhs(i,1) - au(i,1)
        res(i,1) = temp
        u(i,1) = u(i,1) + temp*factor
      else if (id == n_x) then
        au(n_x,1) = (2._RK*u(n_x,1)-u(n_x-1,1))*invdx2 &
                  + (2._RK*u(n_x,1)-u(n_x,2))*invdy2
        temp = rhs(n_x,1) - au(n_x,1)
        res(n_x,1) = temp
        u(n_x,1) = u(n_x,1) + temp*factor
      else if (id == n_x+1) then
        au(1,n_y) = (2._RK*u(1,n_y)-u(2,1))*invdx2 &
                  + (2._RK*u(1,n_y)-u(1,n_y-1))*invdy2
        temp = rhs(1,n_y) - au(1,n_y)
        res(1,n_y) = temp
        u(1,n_y) = u(1,n_y) + temp*factor
      else if (id <= 2*n_x-1) then
        i = id - n_x
        au(i,n_y) = (-u(i-1,n_y)+2._RK*u(i,n_y)-u(i+1,n_y))*invdx2 &
                  + (2._RK*u(i,n_y)-u(i,n_y-1))*invdy2
        temp = rhs(i,n_y) - au(i,n_y)
        res(i,n_y) = temp
        u(i,n_y) = u(i,n_y) + temp*factor
      else if (id == 2*n_x) then
        au(n_x,n_y) = (2._RK*u(n_x,n_y)-u(n_x-1,1))*invdx2 &
                    + (2._RK*u(n_x,n_y)-u(n_x,n_y-1))*invdy2
        temp = rhs(n_x,n_y) - au(n_x,n_y)
        res(n_x,n_y) = temp
        u(n_x,n_y) = u(n_x,n_y) + temp*factor
      else if (id <= 2*n_x+n_y-2) then
        j = id - 2*n_x + 1
        au(1,j) = (2._RK*u(1,j)-u(2,j))*invdx2 &
                + (-u(1,j-1)+2._RK*u(1,j)-u(1,j+1))*invdy2
        temp = rhs(1,j) - au(1,j)
        res(1,j) = temp
        u(1,j) = u(1,j) + temp*factor
      else
        j = id - 2*n_x - n_y + 3
        au(n_x,j) = (2._RK*u(n_x,j)-u(n_x-1,j))*invdx2 &
                  + (-u(n_x,j-1)+2._RK*u(n_x,j)-u(n_x,j+1))*invdy2
        temp = rhs(n_x,j) - au(n_x,j)
        res(n_x,j) = temp
        u(n_x,j) = u(n_x,j) + temp*factor
      end if
    end do

  end subroutine fused_laplacian_update

end module laplacian_mod
