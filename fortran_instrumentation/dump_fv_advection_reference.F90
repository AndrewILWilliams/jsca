!! Standalone driver: golden fixtures for a_grid_horiz_advection (A-grid van Leer
!! horizontal advection on the sphere), produced by the actual Isca Fortran
!! (src/atmos_spectral/model/fv_advection.F90 compiled unmodified from the pinned
!! tree). fms_mod/constants_mod are fms_stubs.F90; mpp_domains_mod is the serial
!! single-PE stub mpp_domains_fv_stub.F90 (full compute domain, no-op halo
!! exchange).
!!
!! Fortran grid storage is (i, j, k) = (lon, lat, level); the port uses
!! (..., nlat, nlon) with level a leading batch axis, so the test moves axes.
!!
!! Cell edges yy are arcsin of equally spaced sin(lat): non-uniform spacing (to
!! exercise dy_plus/dy_minus and dyy weights) with the poles at +/- pi/2. Winds
!! are strong so that cos(lat) -> 0 near the poles pushes |Courant| > 1 and
!! exercises integer_flux_x.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 mpp_domains_fv_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/fv_advection.F90 \
!!     dump_fv_advection_reference.F90 -o dump_fv_advection_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_fv_advection ./dump_fv_advection_reference

program dump_fv_advection_reference
use fv_advection_mod, only: fv_advection_init, a_grid_horiz_advection
use constants_mod, only: pi
use jsca_dump_mod
implicit none

integer, parameter :: nx = 64, ny = 32, nz = 4
real,    parameter :: dt = 1800.0, degrees_lon = 360.0

real :: yy(ny+1), sinv(ny+1), ycen(ny)
real :: ua(nx,ny,nz), va(nx,ny,nz), q(nx,ny,nz)
real :: dqdt_in(nx,ny,nz), dqdt(nx,ny,nz)
real :: r(nx,ny,nz), meta(5), bmax, dx
integer :: i, j, k

! non-uniform cell-edge latitudes: arcsin of equally spaced sin, poles at +-pi/2
do k = 1, ny+1
  sinv(k) = -1.0 + 2.0*real(k-1)/real(ny)
end do
sinv(1)    = -1.0
sinv(ny+1) =  1.0
yy = asin(sinv)

call fv_advection_init(nx, ny, yy, degrees_lon)

! strong random winds (drive |Courant| > 1 near the poles) and a positive tracer
call random_number(r); ua = (r - 0.5)*160.0    ! +/- 80 m/s
call random_number(r); va = (r - 0.5)*60.0     ! +/- 30 m/s
call random_number(r); q  = 1.0 + r            ! ~[1, 2]
call random_number(r); dqdt_in = (r - 0.5)*1.0e-4

! report the largest zonal Courant number (using cell-CENTRE cosine, as the
! scheme does) so the integer-flux path is confirmed exercised (|Courant| > 1)
ycen = 0.5*(yy(2:ny+1) + yy(1:ny))
dx = (degrees_lon/360.0)*2.0*pi*6376.0e3/real(nx)
bmax = 0.0
do k = 1, nz
  do j = 1, ny
    do i = 1, nx
      bmax = max(bmax, abs(ua(i,j,k)*dt/(dx*cos(ycen(j)))))
    end do
  end do
end do
write(*,*) 'max |zonal Courant| (centre) =', bmax

meta(1) = nx; meta(2) = ny; meta(3) = nz; meta(4) = dt; meta(5) = degrees_lon
call jsca_dump_1d('fv_meta', meta)
call jsca_dump_1d('fv_yy', yy)
call jsca_dump_3d('fv_ua', ua)
call jsca_dump_3d('fv_va', va)
call jsca_dump_3d('fv_q', q)
call jsca_dump_3d('fv_dqdt_in', dqdt_in)

! advective form (flux argument absent -> .false.)
dqdt = dqdt_in
call a_grid_horiz_advection(ua, va, q, dt, dqdt)
call jsca_dump_3d('fv_dqdt_adv', dqdt)

! flux form
dqdt = dqdt_in
call a_grid_horiz_advection(ua, va, q, dt, dqdt, flux=.true.)
call jsca_dump_3d('fv_dqdt_flux', dqdt)

write(*,*) 'fv_advection reference fixtures dumped'
end program dump_fv_advection_reference
