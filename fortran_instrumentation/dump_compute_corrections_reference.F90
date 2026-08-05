!! Standalone driver: golden fixtures for compute_corrections (the mass/energy
!! conservation corrections of spectral_dynamics.F90). The routine body is
!! compiled VERBATIM from the pinned source via compute_corrections_wrapper.F90
!! (see its header for the full build/run recipe). Links the real
!! global_integral.F90 + press_and_geopot.F90 + gauss_and_legendre.F90 (as the
!! global_integral fixture does) and the transforms_grid_stub for the area mean.
!!
!! Dry Held-Suarez path: do_mass_correction=.true., do_energy_correction=.true.,
!! do_water_correction=.false.. Fortran (lon,lat,lev); the port uses jsca's
!! (nlat,nlon[,K]) grid convention, so the test transposes lon<->lat.

program dump_compute_corrections_reference
use compute_corrections_mod
use transforms_mod,         only: transforms_grid_stub_init
use press_and_geopot_mod,   only: press_and_geopot_init
use gauss_and_legendre_mod, only: compute_gaussian
use tracer_type_mod,        only: tracer_type
use jsca_dump_mod
implicit none

integer, parameter :: nlon = 32, nlat = 16, nz = 8, n_hem = nlat/2, ntime = 3
real    :: sin_hem(n_hem), wts_hem(n_hem), wts_lat(nlat)
real    :: pk(nz+1), bk(nz+1), p_full(nlon,nlat,nz)
real    :: rnd3(nlon,nlat,nz), rnd2(nlon,nlat), meta(3)
real    :: ln_ps00_in, ts00_in(nz), mean_sp_cur, mean_en_cur
real    :: energy(nlon,nlat,nz)
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

! ---- module-variable environment (would come from spectral_dynamics_mod) ----
ms = 0; me = 10; ns = 0; ne = 11
is = 1; ie = nlon; js = 1; je = nlat
num_levels = nz; num_tracers = 1; nhum = 1
previous = 1; current = 2; future = 3
do_mass_correction = .true.; do_energy_correction = .true.; do_water_correction = .false.
dry_model = .true.; water_correction_limit = 0.0
mean_water_previous = 0.0

allocate(psg(nlon,nlat,ntime), ug(nlon,nlat,nz,ntime), vg(nlon,nlat,nz,ntime), tg(nlon,nlat,nz,ntime))
allocate(ln_ps(0:me,0:ne,ntime), ts(0:me,0:ne,nz,ntime))
allocate(grid_tracers(nlon,nlat,nz,ntime,1), spec_tracers(0:me,0:ne,nz,ntime,1))
psg = 0.0; ug = 0.0; vg = 0.0; tg = 0.0; ln_ps = 0.0; ts = 0.0
grid_tracers = 0.0; spec_tracers = 0.0

! ---- future-level state (index 3) ----
call random_number(rnd2); psg(:,:,future) = 1.0e5*(0.98 + 0.04*rnd2)
call random_number(rnd3); ug(:,:,:,future) = (rnd3 - 0.5)*40.0
call random_number(rnd3); vg(:,:,:,future) = (rnd3 - 0.5)*40.0
call random_number(rnd3); tg(:,:,:,future) = 250.0 + 40.0*rnd3
ln_ps00_in = 11.5                       ! (0,0) coeff of ln(ps)
do k = 1, nz
  ts00_in(k) = sqrt(2.0)*250.0          ! (0,0) coeff of T
end do
ln_ps(0,0,future)  = cmplx(ln_ps00_in, 0.0)
ts(0,0,:,future)   = cmplx(ts00_in, 0.0)

! ---- previous global means: offset from the current state so corrections fire ----
mean_sp_cur = area_weighted_global_mean(psg(:,:,future))
mean_surf_press_previous = mean_sp_cur*1.0005
energy = 0.5*(ug(:,:,:,future)**2 + vg(:,:,:,future)**2) + cp_air*tg(:,:,:,future)
mean_en_cur = mass_weighted_global_integral(energy, psg(:,:,future))
mean_energy_previous = mean_en_cur*0.9995

do k = 1, nz
  p_full(:,:,k) = 0.5*((pk(k)+bk(k)*psg(:,:,future)) + (pk(k+1)+bk(k+1)*psg(:,:,future)))
end do

! ---- dump inputs ----
meta(1) = nlon; meta(2) = nlat; meta(3) = nz
call jsca_dump_1d('cc_meta', meta)
call jsca_dump_1d('cc_pk', pk)
call jsca_dump_1d('cc_bk', bk)
call jsca_dump_1d('cc_wts_lat', wts_lat)
call jsca_dump_2d('cc_psg_in', psg(:,:,future))
call jsca_dump_3d('cc_ug', ug(:,:,:,future))
call jsca_dump_3d('cc_vg', vg(:,:,:,future))
call jsca_dump_3d('cc_tg_in', tg(:,:,:,future))
call jsca_dump_scalar('cc_lnps00_in', real(ln_ps(0,0,future)))
call jsca_dump_1d('cc_ts00_in', real(ts(0,0,:,future)))
call jsca_dump_scalar('cc_mean_sp_prev', mean_surf_press_previous)
call jsca_dump_scalar('cc_mean_en_prev', mean_energy_previous)

! ---- run and dump outputs ----
call compute_corrections(1200.0, tracer_attributes, p_full)

call jsca_dump_2d('cc_psg_out', psg(:,:,future))
call jsca_dump_3d('cc_tg_out', tg(:,:,:,future))
call jsca_dump_scalar('cc_lnps00_out', real(ln_ps(0,0,future)))
call jsca_dump_1d('cc_ts00_out', real(ts(0,0,:,future)))

write(*,*) 'compute_corrections reference fixtures dumped'
end program dump_compute_corrections_reference
