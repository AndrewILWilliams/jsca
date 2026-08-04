!! spec_mpp_mod stub for the spherical-operators fixture driver.
!!
!! spherical.F90 pulls only get_spec_domain from spec_mpp_mod. This stub returns
!! the full serial spectral domain (single PE): ms=0, me=num_fourier, ns=0,
!! ne=num_spherical — spec_mpp_mod's behaviour for one rank.

module spec_mpp_mod
implicit none
private
public :: spec_mpp_stub_init, get_spec_domain

integer, save :: ms = 0, me = 0, ns = 0, ne = 0

contains

subroutine spec_mpp_stub_init(num_fourier, num_spherical)
  integer, intent(in) :: num_fourier, num_spherical
  ms = 0; me = num_fourier; ns = 0; ne = num_spherical
end subroutine spec_mpp_stub_init

subroutine get_spec_domain(ms_out, me_out, ns_out, ne_out)
  integer, intent(out) :: ms_out, me_out, ns_out, ne_out
  ms_out = ms; me_out = me; ns_out = ns; ne_out = ne
end subroutine get_spec_domain

end module spec_mpp_mod
