!! Standalone driver: golden fixtures for the PPM (piecewise-parabolic,
!! finite-volume) vertical advection scheme FINITE_VOLUME_PARABOLIC, produced by
!! the actual Isca Fortran (src/atmos_shared/vert_advection/vert_advection.F90
!! compiled unmodified from the pinned tree). fms_mod/constants_mod are
!! fms_stubs.F90; mpp_mod is mpp_mod_stub.F90 (uncalled diagnostics only).
!!
!! Fortran storage is (i,j,k) = (lon,lat,level) with level LAST = the port's
!! (..., K) column layout, so no axis move is needed.
!!
!! Two Courant regimes are covered so the fixture exercises BOTH the single-cell
!! PPM flux and the multi-cell Courant>1 extension (the do-while accumulation):
!!   * the first half of the columns use gentle winds of BOTH signs (|Courant|<1)
!!   * the second half use strong, POSITIVE-only winds (Courant up to ~8), driving
!!     the Courant>1 walk over several cells.
!! The strong winds are positive-only on purpose: Isca's PPM has an out-of-bounds
!! bug in the *negative*-wind Courant>1 branch -- its walk exits on `kk==ks` (F90
!! L414) while incrementing kk toward `ke`, so kk runs past ke and reads dz(ke+1)
!! out of bounds (garbage under -O2). The positive branch exits correctly on
!! `kk==ks` (F90 L387). We therefore validate the well-defined paths (all Courant
!! for w>=0, plus |Courant|<1 for w<0); the jsca port clamps the departure cell at
!! ke (the obviously-intended behaviour) and documents the Isca bug. Frierson's
!! resolved vertical Courant number stays below 1, so the buggy path is not
!! exercised there. Both ADVECTIVE_FORM and FLUX_FORM are dumped; the
!! Colella-Woodward limiter is exercised by a non-monotone tracer profile.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 mpp_mod_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_shared/vert_advection/vert_advection.F90 \
!!     dump_ppm_advection_reference.F90 -o dump_ppm_advection_reference
!! Run: JSCA_DUMP_DIR=<dir> ./dump_ppm_advection_reference

program dump_ppm_advection_reference
use vert_advection_mod
use jsca_dump_mod
implicit none

integer, parameter :: nx = 8, ny = 4, nz = 12
real,    parameter :: dt = 600.0

real :: w(nx,ny,nz+1), dz(nx,ny,nz), r(nx,ny,nz), rdt(nx,ny,nz)
real :: rr(nx,ny,nz+1), rnd(nx,ny,nz), meta(4), cn, cnmax
integer :: i, j, k

! layer depths ~ [4000, 8000] Pa (modest, so strong winds reach Courant > 1)
call random_number(rnd); dz = 4000.0 + 4000.0*rnd
! non-monotone tracer profile (exercises the Colella-Woodward limiter)
call random_number(rnd); r = 0.005 + 0.02*rnd

! interface velocities: gentle for the first half of columns (Courant < 1),
! strong for the second half (Courant up to ~8), both signs.
call random_number(rr)
do k = 1, nz+1
  do j = 1, ny
    do i = 1, nx
      if (i <= nx/2) then
        w(i,j,k) = (rr(i,j,k) - 0.5)*8.0     ! ~[-4, 4] Pa/s -> |Courant| < 1, both signs
      else
        w(i,j,k) = rr(i,j,k)*60.0            ! ~[0, 60] Pa/s -> Courant up to ~8, w >= 0 only
      end if
    end do
  end do
end do
w(:,:,1)    = 0.0      ! zero flux at the top boundary interface
w(:,:,nz+1) = 0.0      ! zero flux at the surface boundary interface

! report the largest interior Courant number actually present
cnmax = 0.0
do k = 2, nz
  do j = 1, ny
    do i = 1, nx
      if (w(i,j,k) >= 0.0) then
        cn = dt*w(i,j,k)/dz(i,j,k-1)
      else
        cn = -dt*w(i,j,k)/dz(i,j,k)
      end if
      cnmax = max(cnmax, cn)
    end do
  end do
end do

meta(1) = nx; meta(2) = ny; meta(3) = nz; meta(4) = dt
call jsca_dump_1d('ppm_meta', meta)
call jsca_dump_3d('ppm_w', w)
call jsca_dump_3d('ppm_dz', dz)
call jsca_dump_3d('ppm_r', r)

call vert_advection(dt, w, dz, r, rdt, scheme=FINITE_VOLUME_PARABOLIC, form=ADVECTIVE_FORM)
call jsca_dump_3d('ppm_adv', rdt)
call vert_advection(dt, w, dz, r, rdt, scheme=FINITE_VOLUME_PARABOLIC, form=FLUX_FORM)
call jsca_dump_3d('ppm_flux', rdt)

write(*,*) 'PPM vert_advection reference fixtures dumped; max interior Courant =', cnmax
end program dump_ppm_advection_reference
