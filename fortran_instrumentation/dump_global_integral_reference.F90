!! Standalone driver: golden fixtures for global_integral (mass-weighted vertical
!! integral, area-averaged over the globe), produced by the actual Isca Fortran
!! (src/atmos_spectral/model/global_integral.F90 compiled unmodified from the
!! pinned tree, linked against the real press_and_geopot.F90 and the real
!! gauss_and_legendre.F90 for the Gaussian weights). transforms_mod is stubbed
!! (transforms_grid_stub.F90): area_weighted_global_mean replicates
!! transforms.F90's formula verbatim using the real compute_gaussian weights.
!! mpp_domains_mod is a no-op stub (mpp_global_field is imported but not called).
!!
!! Fortran grid storage is (i, j, k) = (lon, lat, level); the port keeps
!! (lat, lon, level), so the test transposes lon<->lat.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 mpp_domains_stub.F90 transforms_grid_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/tools/gauss_and_legendre.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/press_and_geopot.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/global_integral.F90 \
!!     dump_global_integral_reference.F90 -o dump_global_integral_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_global_integral ./dump_global_integral_reference

program dump_global_integral_reference
use global_integral_mod,   only: mass_weighted_global_integral
use press_and_geopot_mod,  only: press_and_geopot_init, press_and_geopot_end
use transforms_mod,        only: transforms_grid_stub_init
use gauss_and_legendre_mod, only: compute_gaussian
use jsca_dump_mod
implicit none

integer, parameter :: n_hem = 4, nlat = 8, nlon = 16, nlev = 5
real :: sin_hem(n_hem), wts_hem(n_hem), wts_lat(nlat)
real :: pk(nlev+1), bk(nlev+1), field(nlon,nlat,nlev), ps(nlon,nlat)
real :: integral, wr(nlon,nlat,nlev), meta(3)
integer :: j, k

! real Gaussian weights, assembled south -> north exactly as jsca.grid.gaussian_grid
call compute_gaussian(sin_hem, wts_hem, n_hem)
do j = 1, n_hem
  wts_lat(j)         = wts_hem(j)             ! southern hemisphere
  wts_lat(n_hem + j) = wts_hem(n_hem + 1 - j) ! northern hemisphere
end do
call transforms_grid_stub_init(nlon, nlat, wts_lat)

! sigma vertical coordinate (pk = 0)
do k = 1, nlev+1
  pk(k) = 0.0
  bk(k) = real(k-1)/real(nlev)
end do
call press_and_geopot_init(pk, bk, .false., 'simmons_and_burridge')

call random_number(wr);  field = 200.0 + 100.0*wr      ! temperature-like field
call random_number(ps);  ps = 1.0e5*(0.95 + 0.1*ps)

integral = mass_weighted_global_integral(field, ps)

meta = (/ real(nlon), real(nlat), real(nlev) /)
call jsca_dump_1d('gi_meta', meta)
call jsca_dump_1d('gi_pk', pk)
call jsca_dump_1d('gi_bk', bk)
call jsca_dump_1d('gi_wts_lat', wts_lat)
call jsca_dump_3d('gi_field', field)
call jsca_dump_2d('gi_ps', ps)
call jsca_dump_scalar('gi_integral', integral)

call press_and_geopot_end()
write(*,*) 'global_integral reference fixtures dumped'
end program dump_global_integral_reference
