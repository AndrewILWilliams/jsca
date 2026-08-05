!! Standalone driver: golden fixtures for vert_advection (vertical advection of
!! grid-column fields), produced by the actual Isca Fortran
!! (src/atmos_shared/vert_advection/vert_advection.F90 compiled unmodified from
!! the pinned tree). fms_mod/constants_mod are fms_stubs.F90; mpp_mod is the
!! serial no-op stub mpp_mod_stub.F90 (used only by the uncalled diagnostics).
!!
!! Fortran storage is (i, j, k) = (lon, lat, level) with level LAST, which is
!! exactly the port's (..., K) column layout, so no axis move is needed.
!!
!! Covers the five jit-safe schemes (2nd/4th centered +/- wts, van Leer linear)
!! in both ADVECTIVE_FORM and FLUX_FORM. Random dz/r/w exercise the slope_z
!! limiter and the unequal-spacing weights.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 mpp_mod_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_shared/vert_advection/vert_advection.F90 \
!!     dump_vert_advection_reference.F90 -o dump_vert_advection_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_vert_advection ./dump_vert_advection_reference

program dump_vert_advection_reference
use vert_advection_mod
use jsca_dump_mod
implicit none

integer, parameter :: nx = 6, ny = 5, nz = 10
real,    parameter :: dt = 600.0

real :: w(nx,ny,nz+1), dz(nx,ny,nz), r(nx,ny,nz), rdt(nx,ny,nz)
real :: rr(nx,ny,nz+1), meta(4)
integer :: schemes(5), s
character(len=4) :: names(5)

schemes = (/ SECOND_CENTERED, SECOND_CENTERED_WTS, FOURTH_CENTERED, FOURTH_CENTERED_WTS, &
             FINITE_VOLUME_LINEAR /)
names   = (/ '2c  ', '2cw ', '4c  ', '4cw ', 'fvl ' /)

call random_number(dz); dz = 5000.0 + 10000.0*dz     ! layer depth ~ [5000, 15000] Pa
call random_number(r);  r  = 220.0 + 100.0*r          ! temperature-like ~ [220, 320]
call random_number(rr); w  = (rr - 0.5)*16.0          ! interface velocity ~ [-8, 8] Pa/s

meta(1) = nx; meta(2) = ny; meta(3) = nz; meta(4) = dt
call jsca_dump_1d('va_meta', meta)
call jsca_dump_3d('va_w', w)
call jsca_dump_3d('va_dz', dz)
call jsca_dump_3d('va_r', r)

do s = 1, 5
  call vert_advection(dt, w, dz, r, rdt, scheme=schemes(s), form=ADVECTIVE_FORM)
  call jsca_dump_3d('va_'//trim(names(s))//'_adv', rdt)
  call vert_advection(dt, w, dz, r, rdt, scheme=schemes(s), form=FLUX_FORM)
  call jsca_dump_3d('va_'//trim(names(s))//'_flux', rdt)
end do

write(*,*) 'vert_advection reference fixtures dumped'
end program dump_vert_advection_reference
