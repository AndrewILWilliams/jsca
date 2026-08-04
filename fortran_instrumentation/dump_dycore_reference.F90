!! Standalone driver: golden fixtures for the time-stepping spine, produced by
!! the actual Isca Fortran (leapfrog.F90, matrix_invert.F90,
!! press_and_geopot.F90 compiled unmodified from the pinned tree).
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/leapfrog.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/matrix_invert.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/press_and_geopot.F90 \
!!     dump_dycore_reference.F90 -o dump_dycore_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_dycore ./dump_dycore_reference

program dump_dycore_reference
use leapfrog_mod, only: leapfrog, leapfrog_2level_A, leapfrog_2level_B
use matrix_invert_mod, only: invert
use press_and_geopot_mod
use jsca_dump_mod
implicit none

integer, parameter :: nm = 6, nn = 5, nl = 4, nt = 3
integer, parameter :: nx = 4, ny = 3, K = 25
real, parameter :: delta_t = 720.0, robert = 0.03, raw = 0.53

complex :: a(nm,nn,nl,nt), a_in(nm,nn,nl,nt), dta(nm,nn,nl), part(nm,nn,nl)
real :: wr1(nm,nn,nl,nt), wr2(nm,nn,nl,nt), wr3(nm,nn,nl), wr4(nm,nn,nl)
real :: ar(nm,nn,nl,nt), ar_in(nm,nn,nl,nt), dtar(nm,nn,nl), partr(nm,nn,nl)

real :: m25(K,K), m25_in(K,K), det25
real :: m4(4,4), m4_in(4,4), det4

real :: pk(K+1), bk(K+1), pk2(K+1), bk2(K+1)
real :: ps(nx,ny), phis(nx,ny), t3(nx,ny,K), q3(nx,ny,K)
real :: p_half(nx,ny,K+1), ln_p_half(nx,ny,K+1), p_full(nx,ny,K), ln_p_full(nx,ny,K)
real :: gfull(nx,ny,K), ghalf(nx,ny,K+1), zbot(nx,ny)
real :: zfull(nx,ny,K), zhalf(nx,ny,K+1)
integer :: kk

include 'pkbk_t42l25.inc'

!=============================== leapfrog =====================================
call random_number(wr1); call random_number(wr2)
a_in = cmplx(wr1 - 0.5, wr2 - 0.5, kind=kind(1.0))
call random_number(wr3); call random_number(wr4)
dta = cmplx(wr3 - 0.5, wr4 - 0.5, kind=kind(1.0))

call jsca_dump_4d('lf_a_in_re', real(a_in)); call jsca_dump_4d('lf_a_in_im', aimag(a_in))
call jsca_dump_3d('lf_dta_re', real(dta));   call jsca_dump_3d('lf_dta_im', aimag(dta))
call jsca_dump_1d('lf_scalars', (/delta_t, robert, raw/))

! combined leapfrog, normal step (previous, current, future) = (1, 2, 3)
a = a_in
call leapfrog(a, dta, 1, 2, 3, delta_t, robert, raw)
call jsca_dump_4d('lf_combined_out_re', real(a)); call jsca_dump_4d('lf_combined_out_im', aimag(a))

! combined leapfrog, first step (1, 1, 2)
a = a_in
call leapfrog(a, dta, 1, 1, 2, delta_t, robert, raw)
call jsca_dump_4d('lf_first_out_re', real(a)); call jsca_dump_4d('lf_first_out_im', aimag(a))

! two-level split with a fake semi-implicit correction between A and B
a = a_in
call leapfrog_2level_A(a, dta, 1, 2, 3, delta_t, robert, raw, part)
call jsca_dump_4d('lf_2lvA_out_re', real(a)); call jsca_dump_4d('lf_2lvA_out_im', aimag(a))
call jsca_dump_3d('lf_2lvA_part_re', real(part)); call jsca_dump_3d('lf_2lvA_part_im', aimag(part))
a(:,:,:,3) = a(:,:,:,3) * 1.01   ! stand-in for the implicit correction
call leapfrog_2level_B(a, part, 2, 3, robert, raw)
call jsca_dump_4d('lf_2lvB_out_re', real(a)); call jsca_dump_4d('lf_2lvB_out_im', aimag(a))

! real-variant two-level A (grid tracers path; note the Fortran quirk)
call random_number(ar_in); call random_number(dtar)
call jsca_dump_4d('lf_r_a_in', ar_in); call jsca_dump_3d('lf_r_dta', dtar)
ar = ar_in
call leapfrog_2level_A(ar, dtar, 1, 2, 3, delta_t, robert, raw, partr)
call jsca_dump_4d('lf_r_2lvA_out', ar); call jsca_dump_3d('lf_r_2lvA_part', partr)

!=============================== matrix_invert ================================
call random_number(m25_in)
do kk = 1, K
  m25_in(kk,kk) = m25_in(kk,kk) + 5.0   ! diagonally dominant, like the SI matrices
end do
call jsca_dump_2d('mi_in_25', m25_in)
m25 = m25_in
call invert(m25, det25)
call jsca_dump_2d('mi_inv_25', m25); call jsca_dump_scalar('mi_det_25', det25)

call random_number(m4_in)
do kk = 1, 4
  m4_in(kk,kk) = m4_in(kk,kk) + 2.0
end do
call jsca_dump_2d('mi_in_4', m4_in)
m4 = m4_in
call invert(m4, det4)
call jsca_dump_2d('mi_inv_4', m4); call jsca_dump_scalar('mi_det_4', det4)

!==================== press_and_geopot: set 1 (Isca T42L25) ===================
! uneven_sigma coefficients from the pinned held_suarez baseline run;
! dry (use_virtual_temperature = .false.), simmons_and_burridge.
call press_and_geopot_init(pk, bk, .false., 'simmons_and_burridge')
call jsca_dump_1d('pg_pk', pk); call jsca_dump_1d('pg_bk', bk)

call random_number(ps);  ps = 1.0e5*(0.95 + 0.1*ps)
call random_number(phis); phis = 9.8*2000.0*phis    ! nonzero topography
call random_number(t3);  t3 = 220.0 + 80.0*t3
call jsca_dump_2d('pg1_ps', ps); call jsca_dump_2d('pg1_phis', phis)
call jsca_dump_3d('pg1_t', t3)

call pressure_variables(p_half, ln_p_half, p_full, ln_p_full, ps)
call jsca_dump_3d('pg1_p_half', p_half);   call jsca_dump_3d('pg1_ln_p_half', ln_p_half)
call jsca_dump_3d('pg1_p_full', p_full);   call jsca_dump_3d('pg1_ln_p_full', ln_p_full)

call compute_geopotential(t3, ln_p_half, ln_p_full, phis, gfull, ghalf)
call jsca_dump_3d('pg1_geopot_full', gfull); call jsca_dump_3d('pg1_geopot_half', ghalf)

call compute_z_bot(ps, t3(:,:,K), zbot)
call jsca_dump_2d('pg1_z_bot', zbot)

call compute_pressures_and_heights(t3, ps, phis, zfull, zhalf, p_full, p_half)
call jsca_dump_3d('pg1_z_full', zfull); call jsca_dump_3d('pg1_z_half', zhalf)

!============== press_and_geopot: set 2 (hybrid top, virtual T) ===============
call press_and_geopot_end()
do kk = 1, K+1
  pk2(kk) = 500.0*real(K+1-kk)/real(K)          ! nonzero model-top pressure
  bk2(kk) = (real(kk-1)/real(K))**1.5
end do
call press_and_geopot_init(pk2, bk2, .true., 'simmons_and_burridge')
call jsca_dump_1d('pg2_pk', pk2); call jsca_dump_1d('pg2_bk', bk2)

call random_number(q3); q3 = 0.02*q3
call jsca_dump_3d('pg2_q', q3)

call pressure_variables(p_half, ln_p_half, p_full, ln_p_full, ps)
call jsca_dump_3d('pg2_p_half', p_half);   call jsca_dump_3d('pg2_ln_p_half', ln_p_half)
call jsca_dump_3d('pg2_p_full', p_full);   call jsca_dump_3d('pg2_ln_p_full', ln_p_full)

call compute_geopotential(t3, ln_p_half, ln_p_full, phis, gfull, ghalf, q3)
call jsca_dump_3d('pg2_geopot_full', gfull); call jsca_dump_3d('pg2_geopot_half', ghalf)

call compute_z_bot(ps, t3(:,:,K), zbot, q3(:,:,K))
call jsca_dump_2d('pg2_z_bot', zbot)

write(*,*) 'dycore reference fixtures dumped'
end program dump_dycore_reference
