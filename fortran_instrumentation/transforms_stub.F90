!! transforms_mod stub for the spectral_damping fixture driver.
!!
!! spectral_damping.F90 pulls only two entries from transforms_mod:
!! get_eigen_laplacian and get_spec_domain. This stub supplies both, with no
!! reinvented numerics: get_eigen_laplacian *replicates verbatim* the eigenvalue
!! construction of the source it wraps,
!! $ISCA_SRC/src/atmos_spectral/tools/spherical.F90 (subroutine spherical_init,
!! lines 180-192):
!!
!!     fourier_wave(m,n)   = m*fourier_inc
!!     spherical_wave(m,n) = fourier_wave(m,n) + n           ! total wavenumber l
!!     eigen_laplacian     = spherical_wave*(spherical_wave + 1.0)/(radius*radius)
!!
!! i.e. eigen(m,n) = l(l+1)/a^2 with l = m*fourier_inc + n, a POSITIVE quantity
!! (spherical.F90 negates it, `factor = -eigen_laplacian`, to form the actual
!! Laplacian in compute_laplacian). get_spec_domain returns the full serial
!! spectral domain (single PE), which is spec_mpp_mod's behaviour for one rank.

module transforms_mod
implicit none
private
public :: transforms_stub_init, get_eigen_laplacian, get_spec_domain

integer, save :: num_fourier = 0, num_spherical = 0, fourier_inc = 1
integer, save :: ms = 0, me = 0, ns = 0, ne = 0
real,    save :: radius = 0.0
real, allocatable, save :: eigen_laplacian(:,:)

contains

subroutine transforms_stub_init(radius_in, num_fourier_in, num_spherical_in, fourier_inc_in)
  real,    intent(in) :: radius_in
  integer, intent(in) :: num_fourier_in, num_spherical_in, fourier_inc_in
  integer :: m, n
  real    :: fourier_wave, spherical_wave
  radius        = radius_in
  num_fourier   = num_fourier_in
  num_spherical = num_spherical_in
  fourier_inc   = fourier_inc_in
  ms = 0; me = num_fourier; ns = 0; ne = num_spherical
  if (allocated(eigen_laplacian)) deallocate(eigen_laplacian)
  allocate(eigen_laplacian(0:num_fourier, 0:num_spherical))
  do n = 0, num_spherical
    do m = 0, num_fourier
      fourier_wave   = m*fourier_inc
      spherical_wave = fourier_wave + n
      eigen_laplacian(m,n) = spherical_wave*(spherical_wave + 1.0)/(radius*radius)
    end do
  end do
end subroutine transforms_stub_init

subroutine get_eigen_laplacian(eigen_laplacian_out)
  real, intent(out), dimension(:,:) :: eigen_laplacian_out
  eigen_laplacian_out = eigen_laplacian(ms:me, ns:ne)
end subroutine get_eigen_laplacian

subroutine get_spec_domain(ms_out, me_out, ns_out, ne_out)
  integer, intent(out) :: ms_out, me_out, ns_out, ne_out
  ms_out = ms; me_out = me; ns_out = ns; ne_out = ne
end subroutine get_spec_domain

end module transforms_mod
