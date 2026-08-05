!! Minimal mpp_domains_mod stub for fv_advection.F90 (serial, single-PE build).
!!
!! fv_advection uses: domain2D, mpp_define_domains, mpp_update_domains,
!! mpp_get_compute_domain. On one PE the compute domain is the full grid and
!! the halo exchange is a no-op (the meridional halos are filled by the module's
!! explicit pole-reflection code, and longitude is periodic by index wrapping).
!! No numerics live here.
!!
!! (Separate from mpp_domains_stub.F90, which serves global_integral with a
!! different public interface; do not compile both in one build.)

module mpp_domains_mod
implicit none
private
public :: domain2D, mpp_define_domains, mpp_update_domains, mpp_get_compute_domain

integer, save :: dom_nx = 0, dom_ny = 0

type domain2D
  integer :: nx = 0, ny = 0
end type domain2D

interface mpp_update_domains
  module procedure mpp_update_domains_r3d
end interface

contains

subroutine mpp_define_domains(global_indices, layout, domain, xhalo, yhalo, xflags, yflags, name)
  integer, intent(in) :: global_indices(:)
  integer, intent(in) :: layout(:)
  type(domain2D), intent(inout) :: domain
  integer, intent(in), optional :: xhalo, yhalo, xflags, yflags
  character(len=*), intent(in), optional :: name
  domain%nx = global_indices(2) - global_indices(1) + 1
  domain%ny = global_indices(4) - global_indices(3) + 1
  dom_nx = domain%nx
  dom_ny = domain%ny
end subroutine mpp_define_domains

subroutine mpp_get_compute_domain(domain, is, ie, js, je)
  type(domain2D), intent(in) :: domain
  integer, intent(out) :: is, ie, js, je
  is = 1
  ie = domain%nx
  js = 1
  je = domain%ny
end subroutine mpp_get_compute_domain

subroutine mpp_update_domains_r3d(field, domain)
  real, intent(inout) :: field(:,:,:)
  type(domain2D), intent(in) :: domain
end subroutine mpp_update_domains_r3d

end module mpp_domains_mod
