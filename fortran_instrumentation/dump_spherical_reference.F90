!! Standalone driver: golden fixtures for the spherical-harmonic spectral
!! operators, produced by the actual Isca Fortran
!! (src/atmos_spectral/tools/spherical.F90 compiled unmodified from the pinned
!! tree). spec_mpp_mod is stubbed (spec_mpp_stub.F90) for get_spec_domain;
!! fms_mod is the fms_stubs.F90 stub. radius is passed to spherical_init, so no
!! constants_mod dependency.
!!
!! Fortran spectral storage is (m, n, k) with (m, n) first; the port keeps (m, n)
!! last (transforms.py convention), so the test transposes these fixtures.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 spec_mpp_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/tools/spherical.F90 \
!!     dump_spherical_reference.F90 -o dump_spherical_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_spherical ./dump_spherical_reference

program dump_spherical_reference
use spherical_mod, only: spherical_init, spherical_end, &
        compute_lon_deriv_cos, compute_lat_deriv_cos, compute_gradient_cos, &
        compute_laplacian, compute_ucos_vcos, compute_vor_div, compute_vor, compute_div
use spec_mpp_mod, only: spec_mpp_stub_init
use jsca_dump_mod
implicit none

integer, parameter :: num_fourier = 6, num_spherical = 7, fourier_inc = 1
integer, parameter :: nm = num_fourier + 1, nn = num_spherical + 1, K = 3
real,    parameter :: radius = 6376.0e3

real    :: wr(nm, nn, K), wi(nm, nn, K)
complex :: spec(nm, nn, K), vor(nm, nn, K), div(nm, nn, K)
complex :: ucos(nm, nn, K), vcos(nm, nn, K)
complex :: dlon(nm, nn, K), dlat(nm, nn, K)
complex :: lap1(nm, nn, K), lap2(nm, nn, K)
complex :: vor2(nm, nn, K), div2(nm, nn, K), vorf(nm, nn, K), divf(nm, nn, K)
real    :: meta(5)

call spec_mpp_stub_init(num_fourier, num_spherical)
call spherical_init(radius, num_fourier, fourier_inc, num_spherical)

meta = (/ dble(num_fourier), dble(num_spherical), dble(fourier_inc), dble(K), radius /)
call jsca_dump_1d('sph_meta', meta)

! random complex spectral fields
call random_number(wr); call random_number(wi)
spec = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
call random_number(wr); call random_number(wi)
vor = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
call random_number(wr); call random_number(wi)
div = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
call random_number(wr); call random_number(wi)
ucos = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
call random_number(wr); call random_number(wi)
vcos = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))

call dump3('sph_spec', spec)
call dump3('sph_vor', vor)
call dump3('sph_div', div)
call dump3('sph_ucos', ucos)
call dump3('sph_vcos', vcos)

! longitude / latitude derivatives (cos-weighted)
dlon = compute_lon_deriv_cos(spec)
call dump3('sph_dlon', dlon)
dlat = compute_lat_deriv_cos(spec)
call dump3('sph_dlat', dlat)
! gradient (should equal dlon, dlat)
call compute_gradient_cos(spec, dlon, dlat)
call dump3('sph_grad_lon', dlon)
call dump3('sph_grad_lat', dlat)

! laplacian (default = -eigen) and squared (power=2)
lap1 = compute_laplacian(spec)
call dump3('sph_lap1', lap1)
lap2 = compute_laplacian(spec, 2)
call dump3('sph_lap2', lap2)

! (vor, div) -> (u cos, v cos)
call compute_ucos_vcos(vor, div, ucos, vcos)
call dump3('sph_ucos_out', ucos)
call dump3('sph_vcos_out', vcos)

! (u cos, v cos) -> (vor, div)  [independent random ucos/vcos re-read below]
call random_number(wr); call random_number(wi)
ucos = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
call random_number(wr); call random_number(wi)
vcos = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
call dump3('sph_ucos2', ucos)
call dump3('sph_vcos2', vcos)
call compute_vor_div(ucos, vcos, vor2, div2)
call dump3('sph_vor_div_vor', vor2)
call dump3('sph_vor_div_div', div2)
vorf = compute_vor(ucos, vcos)
divf = compute_div(ucos, vcos)
call dump3('sph_vor_out', vorf)
call dump3('sph_div_out', divf)

call spherical_end()
write(*,*) 'spherical reference fixtures dumped'

contains

subroutine dump3(name, z)
  character(len=*), intent(in) :: name
  complex, intent(in) :: z(:,:,:)
  call jsca_dump_3d(name//'_re', real(z)); call jsca_dump_3d(name//'_im', aimag(z))
end subroutine dump3

end program dump_spherical_reference
