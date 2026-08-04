!! Standalone driver: golden fixtures for spectral_damping, produced by the
!! actual Isca Fortran (src/atmos_spectral/model/spectral_damping.F90 compiled
!! unmodified from the pinned tree). transforms_mod is stubbed
!! (transforms_stub.F90) so get_eigen_laplacian/get_spec_domain resolve without
!! the full spectral-transform graph; the stub replicates spherical.F90's own
!! eigenvalue formula (see its header).
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 transforms_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/spectral_damping.F90 \
!!     dump_spectral_damping_reference.F90 -o dump_spectral_damping_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_spectral_damping ./dump_spectral_damping_reference
!!
!! The three damping_option branches are exercised in separate init scenarios,
!! with spectral_damping_end between them. Note: spectral_damping_end does NOT
!! reset the module's sticky damping_option_exponential flag, so the exponential
!! scenario is run last (see spectral_damping.F90 L45/L136 and the port docstring).

program dump_spectral_damping_reference
use spectral_damping_mod, only: spectral_damping_init, spectral_damping_end, &
        compute_spectral_damping, compute_spectral_damping_vor, compute_spectral_damping_div
use transforms_mod, only: transforms_stub_init, get_eigen_laplacian
use jsca_dump_mod
implicit none

integer, parameter :: num_fourier = 42, num_spherical = 43, fourier_inc = 1
integer, parameter :: nm = num_fourier + 1, nn = num_spherical + 1, K = 5
real,    parameter :: radius = 6376.0e3, current_dt = 600.0

real    :: eigen(nm, nn)
real    :: wr(nm, nn, K), wi(nm, nn, K)
complex :: f3(nm, nn, K), g3(nm, nn, K)
complex :: og3(nm, nn, K), ovor(nm, nn, K), odiv(nm, nn, K)
complex :: f2(nm, nn), g2(nm, nn), og2(nm, nn)
real    :: meta(5)

call transforms_stub_init(radius, num_fourier, num_spherical, fourier_inc)
call get_eigen_laplacian(eigen)
call jsca_dump_2d('sd_eigen', eigen)
meta = (/ dble(num_fourier), dble(num_spherical), dble(fourier_inc), dble(K), current_dt /)
call jsca_dump_1d('sd_meta', meta)

! Shared random complex inputs (state f3 / tendency g3) reused across scenarios.
call random_number(wr); call random_number(wi)
f3 = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
call random_number(wr); call random_number(wi)
g3 = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
f2 = f3(:,:,1); g2 = g3(:,:,1)
call jsca_dump_3d('sd_f3_re', real(f3)); call jsca_dump_3d('sd_f3_im', aimag(f3))
call jsca_dump_3d('sd_g3_re', real(g3)); call jsca_dump_3d('sd_g3_im', aimag(g3))

!====================== scenario RD: resolution_dependent =====================
call spectral_damping_init(1.157e-04, 4, 'resolution_dependent', 15, &
        num_fourier, num_spherical, K, 5.0e-05, 1.5e-04, 2.5e-04,    &
        damping_coeff_vor=2.3e-04, damping_order_vor=4,              &
        damping_coeff_div=3.1e-04, damping_order_div=2,              &
        damping_coeff_r=1.0e-06)
call run_scenario('rd')
call spectral_damping_end()

!===================== scenario RI: resolution_independent ====================
call spectral_damping_init(1.0e18, 2, 'resolution_independent', 15, &
        num_fourier, num_spherical, K, 5.0e-05, 1.5e-04, 2.5e-04,   &
        damping_coeff_vor=2.0e18, damping_order_vor=2,              &
        damping_coeff_div=1.5e12, damping_order_div=1)
call run_scenario('ri')
call spectral_damping_end()

!====================== scenario EXP: exponential_cutoff ======================
! run last: the module's damping_option_exponential flag is sticky across end().
call spectral_damping_init(1.0e-04, 2, 'exponential_cutoff', 15,   &
        num_fourier, num_spherical, K, 5.0e-05, 1.5e-04, 2.5e-04,  &
        damping_coeff_vor=2.0e-04, damping_order_vor=3,            &
        damping_coeff_div=3.0e-04, damping_order_div=1)
call run_scenario('exp')
call spectral_damping_end()

write(*,*) 'spectral_damping reference fixtures dumped'

contains

subroutine run_scenario(tag)
  character(len=*), intent(in) :: tag
  ! generic 3d
  og3 = g3
  call compute_spectral_damping(f3, og3, current_dt)
  call jsca_dump_3d('sd_'//tag//'_gen3_re', real(og3))
  call jsca_dump_3d('sd_'//tag//'_gen3_im', aimag(og3))
  ! generic 2d
  og2 = g2
  call compute_spectral_damping(f2, og2, current_dt)
  call jsca_dump_2d('sd_'//tag//'_gen2_re', real(og2))
  call jsca_dump_2d('sd_'//tag//'_gen2_im', aimag(og2))
  ! vorticity (bulk + eddy/zmu sponge on level 1)
  ovor = g3
  call compute_spectral_damping_vor(f3, ovor, current_dt)
  call jsca_dump_3d('sd_'//tag//'_vor_re', real(ovor))
  call jsca_dump_3d('sd_'//tag//'_vor_im', aimag(ovor))
  ! divergence (bulk + eddy/zmv sponge on level 1)
  odiv = g3
  call compute_spectral_damping_div(f3, odiv, current_dt)
  call jsca_dump_3d('sd_'//tag//'_div_re', real(odiv))
  call jsca_dump_3d('sd_'//tag//'_div_im', aimag(odiv))
end subroutine run_scenario

end program dump_spectral_damping_reference
