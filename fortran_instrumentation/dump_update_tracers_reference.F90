!! Standalone driver: golden fixtures for the GRID branch of update_tracers (the
!! moist tracer time-step of spectral_dynamics.F90). The grid-branch statements are
!! compiled VERBATIM from the pinned source via update_tracers_wrapper.F90 (see its
!! header for the full build/run recipe), calling the REAL, unmodified
!! a_grid_horiz_advection (fv_advection.F90) and vert_advection (vert_advection.F90).
!!
!! Frierson tracer settings (Isca field_table): a single grid tracer sphum,
!! robert_filter='on' (robert_coeff=0.03), hole_filling='off' (so the grid branch
!! never calls water_borrowing). We drive the NON-last sub-step
!! (step_number /= num_steps): the branch that computes the future-level RAW
!! correction (F90 L1243) which the trailing L1248 assignment then overwrites --
!! the RAW dead-store quirk this fixture pins down.
!!
!! Vertical scheme: this fixture validates the tracer TIME-STEP ASSEMBLY (physics
!! tendency + advection wiring + Robert/RAW filter), which is scheme-agnostic, so
!! it uses SECOND_CENTERED -- a scheme jsca's vert_advection already ports.
!! Frierson's production scheme is finite_volume_parabolic (PPM), whose jsca
!! vert_advection port is the next follow-up; update_grid_tracer takes the scheme
!! as an argument and needs no change once PPM lands.
!!
!! Fortran grid storage is (i,j,k) = (lon,lat,level); the port uses (nlat,nlon,K),
!! so the test transposes lon<->lat.

program dump_update_tracers_reference
use update_tracers_mod
use fv_advection_mod,   only: fv_advection_init
use vert_advection_mod, only: SECOND_CENTERED
use tracer_type_mod,    only: tracer_type
use constants_mod,      only: pi
use jsca_dump_mod
implicit none

integer, parameter :: nx = 32, ny = 16, nz = 8, ntime = 3
real,    parameter :: dt = 720.0, degrees_lon = 360.0
real    :: yy(ny+1), sinv(ny+1)
real    :: wg(nx,ny,nz+1), p_half(nx,ny,nz+1), dt_tr(nx,ny,nz,1)
real    :: part_filt(nx,ny,nz,1), rnd(nx,ny,nz), rnd2(nx,ny), meta(4)
real    :: bk(nz+1), ps
type(tracer_type) :: tracer_attributes(1)
integer :: i, j, k

! module-variable environment (would come from spectral_dynamics_mod)
is = 1; ie = nx; js = 1; je = ny; num_levels = nz; num_tracers = 1
previous = 1; current = 2; future = 3
step_number = 1; num_steps = 4            ! non-last sub-step (exercises the RAW branch)
raw_filter_coeff = 0.53                    ! RAW filter alpha
allocate(tracer_vert_advect_scheme(1)); tracer_vert_advect_scheme(1) = SECOND_CENTERED
allocate(ug(nx,ny,nz,ntime), vg(nx,ny,nz,ntime), grid_tracers(nx,ny,nz,ntime,1))
tracer_attributes(1)%numerical_representation = 'grid'
tracer_attributes(1)%robert_coeff = 0.03   ! Frierson robert_coeff

! non-uniform cell-edge latitudes (arcsin of equally spaced sin, poles at +-pi/2)
do k = 1, ny+1
  sinv(k) = -1.0 + 2.0*real(k-1)/real(ny)
end do
sinv(1) = -1.0; sinv(ny+1) = 1.0
yy = asin(sinv)
call fv_advection_init(nx, ny, yy, degrees_lon)

! winds at the current level; humidity at previous/current; physics tendency dt_tr
call random_number(rnd); ug(:,:,:,current) = (rnd - 0.5)*60.0    ! +/- 30 m/s
call random_number(rnd); vg(:,:,:,current) = (rnd - 0.5)*30.0
call random_number(rnd); grid_tracers(:,:,:,previous,1) = 0.001 + 0.02*rnd
call random_number(rnd); grid_tracers(:,:,:,current,1)  = 0.001 + 0.02*rnd
grid_tracers(:,:,:,future,1) = 0.0
call random_number(rnd); dt_tr(:,:,:,1) = (rnd - 0.5)*2.0e-6      ! phys moistening tendency

! pure-sigma half pressures (physical, monotonic); wg interface mass flux
do k = 1, nz+1
  bk(k) = real(k-1)/real(nz)
end do
call random_number(rnd2); ps = 1.0e5
do k = 1, nz+1
  p_half(:,:,k) = bk(k)*(0.98e5 + 0.04e5*rnd2)
end do
call random_number(rnd); wg = 0.0
do k = 2, nz
  wg(:,:,k) = (rnd(:,:,k-1) - 0.5)*2.0       ! interior interface flux; 0 at top/surface
end do

! --- dump inputs ---
meta(1) = nx; meta(2) = ny; meta(3) = nz; meta(4) = dt
call jsca_dump_1d('ut_meta', meta)
call jsca_dump_1d('ut_yy', yy)
call jsca_dump_scalar('ut_robert', tracer_attributes(1)%robert_coeff)
call jsca_dump_scalar('ut_raw', raw_filter_coeff)
call jsca_dump_3d('ut_q_prev', grid_tracers(:,:,:,previous,1))
call jsca_dump_3d('ut_q_cur_in', grid_tracers(:,:,:,current,1))
call jsca_dump_3d('ut_dt_tr', dt_tr(:,:,:,1))
call jsca_dump_3d('ut_ug', ug(:,:,:,current))
call jsca_dump_3d('ut_vg', vg(:,:,:,current))
call jsca_dump_3d('ut_wg', wg)
call jsca_dump_3d('ut_p_half', p_half)

! --- run and dump outputs ---
call update_grid_tracer_ref(tracer_attributes, dt_tr, wg, p_half, dt, part_filt)

call jsca_dump_3d('ut_q_cur_out', grid_tracers(:,:,:,current,1))
call jsca_dump_3d('ut_q_fut_out', grid_tracers(:,:,:,future,1))
call jsca_dump_3d('ut_part_filt', part_filt(:,:,:,1))

write(*,*) 'update_tracers (grid branch) reference fixtures dumped'
end program dump_update_tracers_reference
