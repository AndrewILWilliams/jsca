!! Standalone driver: golden fixtures for hs_forcing (Held & Suarez 1994 idealized
!! forcing), produced by the actual Isca Fortran
!! (src/atmos_param/hs_forcing/hs_forcing.F90 compiled unmodified from the pinned
!! tree). fms_mod/constants_mod are fms_stubs.F90; the rest of the FMS
!! infrastructure is no-op'd by hs_forcing_stubs.F90. Default Held_Suarez path
!! (do_conserve_energy=.true., no tracers, no diagnostics).
!!
!! Fortran storage is (i, j, k) = (lon, lat, level) with level LAST, exactly the
!! port's (..., K) column layout, so no axis move is needed. The dumped forcing
!! is the composite (udt, vdt, tdt) returned by the public hs_forcing (udt/vdt =
!! Rayleigh drag, tdt = Newtonian heating + Rayleigh frictional heating). A sigma
!! grid places several levels in the boundary layer (sigma > sigma_b = 0.7) to
!! exercise the damping where-clause.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 hs_forcing_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_param/hs_forcing/hs_forcing.F90 \
!!     dump_hs_forcing_reference.F90 -o dump_hs_forcing_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_hs_forcing ./dump_hs_forcing_reference

program dump_hs_forcing_reference
use hs_forcing_mod, only: hs_forcing, hs_forcing_init
use transforms_mod, only: transforms_grid_init
use time_manager_mod, only: time_type
use constants_mod, only: pi
use jsca_dump_mod
implicit none

integer, parameter :: nlon = 8, nlat = 6, nlev = 10
real,    parameter :: dt = 600.0

real :: lon(nlon,nlat), lat(nlon,nlat), ps(nlon,nlat)
real :: p_half(nlon,nlat,nlev+1), p_full(nlon,nlat,nlev)
real :: u(nlon,nlat,nlev), v(nlon,nlat,nlev), t(nlon,nlat,nlev)
real :: um(nlon,nlat,nlev), vm(nlon,nlat,nlev), tm(nlon,nlat,nlev)
real :: udt(nlon,nlat,nlev), vdt(nlon,nlat,nlev), tdt(nlon,nlat,nlev)
real :: zfull(nlon,nlat,nlev)
real :: r(nlon,nlat,nlev,0), rm(nlon,nlat,nlev,0), rdt(nlon,nlat,nlev,0)
real :: lonb(nlon+1,nlat+1), latb(nlon+1,nlat+1)
real :: bk(nlev+1), rnd(nlon,nlat,nlev), meta(4)
type(time_type) :: Time
integer :: i, j, k, axes(4)

! grid: latitudes -pi/2..pi/2 (row-dependent), longitudes 0..2pi
do j = 1, nlat
  do i = 1, nlon
    lat(i,j) = -0.5*pi + pi*(real(j)-0.5)/real(nlat)
    lon(i,j) = 2.0*pi*(real(i)-0.5)/real(nlon)
  end do
end do
lonb = 0.0; latb = 0.0   ! only used for tg_prev allocation shape

! surface pressure and a sigma pressure grid (even sigma), so sigma = p_full/ps
call random_number(rnd(:,:,1)); ps = 1.0e5*(0.98 + 0.04*rnd(:,:,1))
do k = 1, nlev+1
  bk(k) = real(k-1)/real(nlev)
  p_half(:,:,k) = bk(k)*ps
end do
do k = 1, nlev
  p_full(:,:,k) = 0.5*(p_half(:,:,k) + p_half(:,:,k+1))
end do

! state: temperature ~250 K, winds ~[-20,20] m/s; previous step slightly offset
call random_number(rnd); t  = 250.0 + 40.0*rnd
call random_number(rnd); u  = (rnd - 0.5)*40.0
call random_number(rnd); v  = (rnd - 0.5)*40.0
call random_number(rnd); um = u + (rnd - 0.5)*2.0
call random_number(rnd); vm = v + (rnd - 0.5)*2.0
tm = t
zfull = 0.0

axes = (/ 1, 2, 3, 4 /)
call transforms_grid_init(nlon, nlat)
call hs_forcing_init(axes, Time, lonb, latb, lat)

udt = 0.0; vdt = 0.0; tdt = 0.0
call hs_forcing(1, nlon, 1, nlat, dt, Time, lon, lat, p_half, p_full, &
                u, v, t, r, um, vm, tm, rm, udt, vdt, tdt, rdt, zfull)

meta(1) = nlon; meta(2) = nlat; meta(3) = nlev; meta(4) = dt
call jsca_dump_1d('hs_meta', meta)
call jsca_dump_2d('hs_lat', lat)
call jsca_dump_3d('hs_p_half', p_half)
call jsca_dump_3d('hs_p_full', p_full)
call jsca_dump_3d('hs_u', u)
call jsca_dump_3d('hs_v', v)
call jsca_dump_3d('hs_t', t)
call jsca_dump_3d('hs_um', um)
call jsca_dump_3d('hs_vm', vm)
call jsca_dump_3d('hs_udt', udt)
call jsca_dump_3d('hs_vdt', vdt)
call jsca_dump_3d('hs_tdt', tdt)

write(*,*) 'hs_forcing reference fixtures dumped'
end program dump_hs_forcing_reference
