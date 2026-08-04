!! Standalone driver: golden fixtures for water_borrowing (the negative-humidity
!! "hole filling" fixer), produced by the actual Isca Fortran
!! (src/atmos_spectral/model/water_borrowing.F90 compiled unmodified from the
!! pinned tree). transforms_mod is stubbed (transforms_grid_stub.F90) for
!! get_grid_domain (full serial grid); fms_mod is the fms_stubs.F90 stub.
!!
!! Fortran grid storage is (i, j, k) = (lon, lat, level); the port keeps
!! (lat, lon, level), so the test transposes lon<->lat.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 transforms_grid_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/water_borrowing.F90 \
!!     dump_water_borrowing_reference.F90 -o dump_water_borrowing_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_water_borrowing ./dump_water_borrowing_reference

program dump_water_borrowing_reference
use water_borrowing_mod, only: water_borrowing
use transforms_mod, only: transforms_grid_stub_init
use jsca_dump_mod
implicit none

integer, parameter :: nlon = 8, nlat = 4, nlev = 5
real,    parameter :: delta_t = 720.0

real :: qg(nlon, nlat, nlev), dt_qg(nlon, nlat, nlev), dt_qg_in(nlon, nlat, nlev)
real :: p_half(nlon, nlat, nlev+1), ps(nlon, nlat), bk(nlev+1)
real :: wr(nlon, nlat, nlev), wts_dummy(nlat), meta(4)
integer :: k

wts_dummy = 1.0
call transforms_grid_stub_init(nlon, nlat, wts_dummy)  ! only get_grid_domain used

! sigma half levels (pk=0): p_half = bk(k)*ps, monotone increasing
do k = 1, nlev+1
  bk(k) = real(k-1)/real(nlev)
end do
call random_number(ps); ps = 1.0e5*(0.95 + 0.1*ps)
do k = 1, nlev+1
  p_half(:,:,k) = bk(k)*ps
end do

! qg with a modest fraction of negatives (holes to fill); dt_qg random
call random_number(wr); qg = wr - 0.2          ! ~20% negative
call random_number(wr); dt_qg = (wr - 0.5)*1.0e-4
dt_qg_in = dt_qg

meta = (/ real(nlon), real(nlat), real(nlev), delta_t /)
call jsca_dump_1d('wb_meta', meta)
call jsca_dump_1d('wb_bk', bk)
call jsca_dump_2d('wb_ps', ps)
call jsca_dump_3d('wb_p_half', p_half)
call jsca_dump_3d('wb_qg', qg)
call jsca_dump_3d('wb_dt_qg_in', dt_qg_in)

! current even (ascending sweep) and current odd (descending) differ only in fp
! accumulation order; the result is otherwise identical.
dt_qg = dt_qg_in
call water_borrowing(dt_qg, qg, 2, p_half, delta_t)
call jsca_dump_3d('wb_dt_qg_even', dt_qg)

dt_qg = dt_qg_in
call water_borrowing(dt_qg, qg, 1, p_half, delta_t)
call jsca_dump_3d('wb_dt_qg_odd', dt_qg)

write(*,*) 'water_borrowing reference fixtures dumped'
end program dump_water_borrowing_reference
