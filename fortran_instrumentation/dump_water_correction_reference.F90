!! Standalone driver: golden fixtures for the WATER (humidity) conservation
!! correction of spectral_dynamics.F90's compute_corrections. The routine body is
!! compiled VERBATIM from the pinned source via compute_corrections_wrapper.F90
!! (see its header for the full build/run recipe); this driver isolates the water
!! branch by turning the mass and energy corrections OFF and running only
!! do_water_correction=.true. with dry_model=.false..
!!
!! Frierson's sphum is a GRID tracer (Isca field_table numerical_representation=
!! 'grid'), so tracer_attributes(nhum)%numerical_representation='grid' and only the
!! grid branch of the water correction runs (the spectral branch is never reached).
!! water_correction_limit = 200 hPa (Frierson value) so the MiMA pressure-limit
!! remapping is genuinely exercised: some levels are above the limit (uncorrected).
!!
!! Fortran storage is (lon,lat,lev); the port uses jsca's (nlat,nlon[,K]) grid
!! convention (global integrals weight by Gaussian latitude), so the test
!! transposes lon<->lat.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   sed -n '1213,1302p' $ISCA_SRC/src/atmos_spectral/model/spectral_dynamics.F90 \
!!       > compute_corrections_body.inc
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 mpp_domains_stub.F90 transforms_grid_stub.F90 tracer_type_stub.F90 \
!!     $ISCA_SRC/src/atmos_spectral/tools/gauss_and_legendre.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/press_and_geopot.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/global_integral.F90 \
!!     compute_corrections_wrapper.F90 jsca_dump.F90 \
!!     dump_water_correction_reference.F90 -o dump_water_correction_reference
!! Run: JSCA_DUMP_DIR=<dir> ./dump_water_correction_reference

program dump_water_correction_reference
use compute_corrections_mod
use transforms_mod,         only: transforms_grid_stub_init
use press_and_geopot_mod,   only: press_and_geopot_init
use gauss_and_legendre_mod, only: compute_gaussian
use global_integral_mod,    only: mass_weighted_global_integral
use tracer_type_mod,        only: tracer_type
use jsca_dump_mod
implicit none

integer, parameter :: nlon = 32, nlat = 16, nz = 8, n_hem = nlat/2, ntime = 3
real    :: sin_hem(n_hem), wts_hem(n_hem), wts_lat(nlat)
real    :: pk(nz+1), bk(nz+1), p_full(nlon,nlat,nz)
real    :: rnd3(nlon,nlat,nz), rnd2(nlon,nlat), meta(3)
real    :: mean_water_cur, sigma
type(tracer_type) :: tracer_attributes(1)
integer :: j, k

! ---- grid setup (Gaussian weights, even-sigma pk/bk) ----
call compute_gaussian(sin_hem, wts_hem, n_hem)
do j = 1, n_hem
  wts_lat(j)         = wts_hem(j)
  wts_lat(n_hem + j) = wts_hem(n_hem + 1 - j)
end do
call transforms_grid_stub_init(nlon, nlat, wts_lat)
do k = 1, nz+1
  pk(k) = 0.0
  bk(k) = real(k-1)/real(nz)
end do
call press_and_geopot_init(pk, bk, .false., 'simmons_and_burridge')

! ---- module-variable environment ----
ms = 0; me = 10; ns = 0; ne = 11
is = 1; ie = nlon; js = 1; je = nlat
num_levels = nz; num_tracers = 1; nhum = 1
previous = 1; current = 2; future = 3
! isolate the water branch: mass/energy OFF, water ON (moist model)
do_mass_correction = .false.; do_energy_correction = .false.; do_water_correction = .true.
dry_model = .false.
water_correction_limit = 200.0e2            ! Frierson: 200 hPa
tracer_attributes(1)%numerical_representation = 'grid'   ! Frierson sphum is a grid tracer

allocate(psg(nlon,nlat,ntime), ug(nlon,nlat,nz,ntime), vg(nlon,nlat,nz,ntime), tg(nlon,nlat,nz,ntime))
allocate(ln_ps(0:me,0:ne,ntime), ts(0:me,0:ne,nz,ntime))
allocate(grid_tracers(nlon,nlat,nz,ntime,1), spec_tracers(0:me,0:ne,nz,ntime,1))
psg = 0.0; ug = 0.0; vg = 0.0; tg = 0.0; ln_ps = 0.0; ts = 0.0
grid_tracers = 0.0; spec_tracers = 0.0

! ---- future-level state ----
call random_number(rnd2); psg(:,:,future) = 1.0e5*(0.98 + 0.04*rnd2)
! humidity: positive, decreasing with height (~ sigma-weighted), plus noise
call random_number(rnd3)
do k = 1, nz
  sigma = 0.5*(bk(k) + bk(k+1))
  grid_tracers(:,:,k,future,nhum) = 0.02*sigma*(0.5 + rnd3(:,:,k))
end do

! full-level pressure (pure sigma here): p_full = 0.5*(p_half(k)+p_half(k+1))
do k = 1, nz
  p_full(:,:,k) = 0.5*((pk(k)+bk(k)*psg(:,:,future)) + (pk(k+1)+bk(k+1)*psg(:,:,future)))
end do

! ---- previous water mean: offset from current so the correction fires ----
mean_water_cur = mass_weighted_global_integral(grid_tracers(:,:,:,future,nhum), psg(:,:,future))
mean_water_previous = mean_water_cur*1.002

! ---- dump inputs ----
meta(1) = nlon; meta(2) = nlat; meta(3) = nz
call jsca_dump_1d('wc_meta', meta)
call jsca_dump_1d('wc_pk', pk)
call jsca_dump_1d('wc_bk', bk)
call jsca_dump_1d('wc_wts_lat', wts_lat)
call jsca_dump_2d('wc_psg', psg(:,:,future))
call jsca_dump_3d('wc_p_full', p_full)
call jsca_dump_3d('wc_q_in', grid_tracers(:,:,:,future,nhum))
call jsca_dump_scalar('wc_mean_water_prev', mean_water_previous)
call jsca_dump_scalar('wc_limit', water_correction_limit)

! ---- run and dump outputs ----
call compute_corrections(1200.0, tracer_attributes, p_full)

call jsca_dump_3d('wc_q_out', grid_tracers(:,:,:,future,nhum))

write(*,*) 'water_correction reference fixtures dumped; q sum in/out ratio', &
     mass_weighted_global_integral(grid_tracers(:,:,:,future,nhum), psg(:,:,future))/mean_water_cur
end program dump_water_correction_reference
