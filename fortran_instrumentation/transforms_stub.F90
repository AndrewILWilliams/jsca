!! transforms_mod stub for the implicit fixture driver.
!!
!! implicit.F90 pulls only get_spec_domain from transforms_mod (eigen and the
!! total-wavenumber table are passed into implicit_init directly). This stub
!! returns the full serial spectral domain (single PE) — spec_mpp_mod's
!! behaviour for one rank: ms=0, me=num_fourier, ns=0, ne=num_spherical.

module transforms_mod
implicit none
private
public :: transforms_stub_init, get_spec_domain

integer, save :: ms = 0, me = 0, ns = 0, ne = 0

contains

subroutine transforms_stub_init(num_fourier, num_spherical)
  integer, intent(in) :: num_fourier, num_spherical
  ms = 0; me = num_fourier; ns = 0; ne = num_spherical
end subroutine transforms_stub_init

subroutine get_spec_domain(ms_out, me_out, ns_out, ne_out)
  integer, intent(out) :: ms_out, me_out, ns_out, ne_out
  ms_out = ms; me_out = me; ns_out = ns; ne_out = ne
end subroutine get_spec_domain

end module transforms_mod
