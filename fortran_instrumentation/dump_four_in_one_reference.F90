!! Standalone driver: golden fixtures for four_in_one (the grid-space
!! pressure-gradient / adiabatic-heating / vertical-velocity / ps-tendency kernel
!! of spectral_dynamics.F90). The routine body is compiled VERBATIM from the
!! pinned source via four_in_one_wrapper.F90 (see its header). fms_stubs.F90
!! supplies constants_mod (rdgas, cp_air).
!!
!! Fortran storage is (i, j, k) = (lon, lat, level) with level LAST, exactly the
!! port's (..., K) column layout. A hybrid coordinate (pk with a nonzero bump,
!! bk even-sigma) exercises the dpk terms; both vert_difference_option branches
!! ('simmons_and_burridge' and 'mcm') are dumped.
!!
!! Build/run: see four_in_one_wrapper.F90 header (add this file to the gfortran
!! line and set JSCA_DUMP_DIR=../tests/fixtures/raw_four_in_one).

program dump_four_in_one_reference
use four_in_one_mod
use constants_mod, only: pi
use jsca_dump_mod
implicit none

integer, parameter :: nlon = 8, nlat = 6, nz = 10
real :: pk(nz+1)
real :: p_surf(nlon,nlat), dx_psg(nlon,nlat), dy_psg(nlon,nlat)
real :: p_half(nlon,nlat,nz+1), ln_p_half(nlon,nlat,nz+1)
real :: p_full(nlon,nlat,nz), ln_p_full(nlon,nlat,nz)
real :: divg(nlon,nlat,nz), u_grid(nlon,nlat,nz), v_grid(nlon,nlat,nz), t_grid(nlon,nlat,nz)
real :: dt_psg_in(nlon,nlat), dt_tg_in(nlon,nlat,nz), dt_ug_in(nlon,nlat,nz), dt_vg_in(nlon,nlat,nz)
real :: dt_psg(nlon,nlat), dt_tg(nlon,nlat,nz), dt_ug(nlon,nlat,nz), dt_vg(nlon,nlat,nz)
real :: wg(nlon,nlat,nz+1), wg_full(nlon,nlat,nz)
real :: rnd3(nlon,nlat,nz), rnd2(nlon,nlat), meta(3)
integer :: k

! module-variable environment (would come from spectral_dynamics_mod)
is = 1; ie = nlon; js = 1; je = nlat; num_levels = nz
allocate(bk(nz+1), dpk(nz), dbk(nz))

! hybrid vertical coordinate: pk a nonzero bump (dpk /= 0), bk even sigma
do k = 1, nz+1
  pk(k) = 100.0 + 3000.0*sin(pi*real(k-1)/real(nz))
  bk(k) = real(k-1)/real(nz)
end do
do k = 1, nz
  dpk(k) = pk(k+1) - pk(k)
  dbk(k) = bk(k+1) - bk(k)
end do

call random_number(rnd2); p_surf = 1.0e5*(0.98 + 0.04*rnd2)
do k = 1, nz+1
  p_half(:,:,k) = pk(k) + bk(k)*p_surf
end do
do k = 1, nz
  p_full(:,:,k) = 0.5*(p_half(:,:,k) + p_half(:,:,k+1))
end do
ln_p_half = log(p_half)
ln_p_full = log(p_full)

call random_number(rnd3); divg   = (rnd3 - 0.5)*1.0e-5
call random_number(rnd3); u_grid = (rnd3 - 0.5)*40.0
call random_number(rnd3); v_grid = (rnd3 - 0.5)*40.0
call random_number(rnd3); t_grid = 250.0 + 40.0*rnd3
call random_number(rnd2); dx_psg = (rnd2 - 0.5)*2.0
call random_number(rnd2); dy_psg = (rnd2 - 0.5)*2.0
call random_number(rnd2); dt_psg_in = (rnd2 - 0.5)*1.0
call random_number(rnd3); dt_tg_in  = (rnd3 - 0.5)*1.0e-3
call random_number(rnd3); dt_ug_in  = (rnd3 - 0.5)*1.0e-3
call random_number(rnd3); dt_vg_in  = (rnd3 - 0.5)*1.0e-3

meta(1) = nlon; meta(2) = nlat; meta(3) = nz
call jsca_dump_1d('fio_meta', meta)
call jsca_dump_1d('fio_pk', pk)
call jsca_dump_1d('fio_bk', bk)
call jsca_dump_2d('fio_p_surf', p_surf)
call jsca_dump_2d('fio_dx_psg', dx_psg)
call jsca_dump_2d('fio_dy_psg', dy_psg)
call jsca_dump_3d('fio_ln_p_half', ln_p_half)
call jsca_dump_3d('fio_ln_p_full', ln_p_full)
call jsca_dump_3d('fio_p_full', p_full)
call jsca_dump_3d('fio_divg', divg)
call jsca_dump_3d('fio_u_grid', u_grid)
call jsca_dump_3d('fio_v_grid', v_grid)
call jsca_dump_3d('fio_t_grid', t_grid)
call jsca_dump_2d('fio_dt_psg_in', dt_psg_in)
call jsca_dump_3d('fio_dt_tg_in', dt_tg_in)
call jsca_dump_3d('fio_dt_ug_in', dt_ug_in)
call jsca_dump_3d('fio_dt_vg_in', dt_vg_in)

call run('simmons_and_burridge', 'sb')
call run('mcm', 'mcm')

write(*,*) 'four_in_one reference fixtures dumped'

contains

  subroutine run(option, tag)
    character(len=*), intent(in) :: option, tag
    vert_difference_option = option
    dt_psg = dt_psg_in; dt_tg = dt_tg_in; dt_ug = dt_ug_in; dt_vg = dt_vg_in
    call four_in_one(divg, u_grid, v_grid, t_grid, p_surf, ln_p_half, ln_p_full, p_full, &
                     dx_psg, dy_psg, dt_psg, wg, wg_full, dt_tg, dt_ug, dt_vg)
    call jsca_dump_2d('fio_'//trim(tag)//'_dt_psg', dt_psg)
    call jsca_dump_3d('fio_'//trim(tag)//'_dt_tg', dt_tg)
    call jsca_dump_3d('fio_'//trim(tag)//'_dt_ug', dt_ug)
    call jsca_dump_3d('fio_'//trim(tag)//'_dt_vg', dt_vg)
    call jsca_dump_3d('fio_'//trim(tag)//'_wg', wg)
    call jsca_dump_3d('fio_'//trim(tag)//'_wg_full', wg_full)
  end subroutine run

end program dump_four_in_one_reference
