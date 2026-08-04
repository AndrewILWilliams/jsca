!! Standalone driver: golden fixtures for implicit (the semi-implicit gravity-wave
!! correction), produced by the actual Isca Fortran
!! (src/atmos_spectral/model/implicit.F90 compiled unmodified from the pinned
!! tree, linked against the real press_and_geopot.F90 and matrix_invert.F90).
!! transforms_mod is stubbed (transforms_stub.F90) to supply get_spec_domain
!! (full serial domain); constants_mod is the fms_stubs.F90 stub.
!!
!! Only implicit_init / implicit_correction / implicit_end are public — the
!! reference-state matrices (div_mat, h, wave_matrix), the linearized
!! temperature/pressure tendency and geopotential, and adjust_dt_divs are all
!! module-private, so they are validated transitively through implicit_correction
!! outputs. Each scenario is a fresh init+correction+end.
!!
!! State note: implicit_end does NOT reset the module's cached dt, and a fresh
!! implicit_init allocates wave_matrix without filling it (build_wave_matrices
!! only fires from implicit_correction when dt changes). So every scenario uses a
!! distinct, nonzero dt to force a wave-matrix rebuild; reusing the previous
!! scenario's dt would read an uninitialised wave_matrix (see the port docstring).
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 transforms_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/matrix_invert.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/press_and_geopot.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/implicit.F90 \
!!     dump_implicit_reference.F90 -o dump_implicit_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_implicit ./dump_implicit_reference

program dump_implicit_reference
use jsca_dump_mod
implicit none

real, parameter :: radius = 6376.0e3

! scenario A1: sigma (pk=0), simmons_and_burridge, dt=600
call run_one('a1', 6, 7, 8, 2, 0.5, 'simmons_and_burridge', .false., 600.0)
! scenario A2: same config, different dt=300 (exercises xi/wave-matrix dt dependence)
call run_one('a2', 6, 7, 8, 2, 0.5, 'simmons_and_burridge', .false., 300.0)
! scenario B: mcm vertical differencing, sigma, dt=450
call run_one('b',  5, 6, 6, 2, 0.6, 'mcm', .false., 450.0)
! scenario C: hybrid (pk/=0), simmons_and_burridge, dt=720
call run_one('c',  5, 6, 6, 2, 0.5, 'simmons_and_burridge', .true., 720.0)

write(*,*) 'implicit reference fixtures dumped'

contains

subroutine run_one(tag, num_fourier, num_spherical, K, ntime, alpha, vopt, use_hybrid, dt)
  use transforms_mod,       only: transforms_stub_init
  use implicit_mod,         only: implicit_init, implicit_correction, implicit_end
  use press_and_geopot_mod, only: press_and_geopot_init, press_and_geopot_end
  character(len=*), intent(in) :: tag, vopt
  integer, intent(in) :: num_fourier, num_spherical, K, ntime
  real,    intent(in) :: alpha, dt
  logical, intent(in) :: use_hybrid

  integer :: nm, nn, num_total, m, n, kk, prev, curr
  real    :: ref_surf_p, sw
  real,    allocatable :: pk(:), bk(:), ref_t(:), eigen(:,:)
  integer, allocatable :: wavenum(:,:)
  real,    allocatable :: wr(:,:,:,:), wi(:,:,:,:)
  complex, allocatable :: divs(:,:,:,:), ts(:,:,:,:), ln_ps(:,:,:)
  complex, allocatable :: dt_divs(:,:,:), dt_ts(:,:,:), dt_ln_ps(:,:)

  nm = num_fourier + 1
  nn = num_spherical + 1
  num_total = num_fourier
  ref_surf_p = 1.0e5
  prev = 1; curr = 2

  call transforms_stub_init(num_fourier, num_spherical)

  allocate(pk(K+1), bk(K+1), ref_t(K), eigen(0:num_fourier,0:num_spherical), &
           wavenum(0:num_fourier,0:num_spherical))

  ! vertical coordinate: half levels k=1(top)..K+1(surface)
  do kk = 1, K+1
    if (use_hybrid) then
      pk(kk) = 2000.0*real(K+1-kk)/real(K)             ! nonzero p_top, decreasing to 0 at surface
      bk(kk) = (real(kk-1)/real(K))**1.2
    else
      pk(kk) = 0.0                                     ! pure sigma
      bk(kk) = (real(kk-1)/real(K))**1.1
    end if
  end do
  ! reference temperature profile (K), a mild lapse
  do kk = 1, K
    ref_t(kk) = 220.0 + 40.0*real(kk-1)/real(K-1)
  end do

  ! eigen(m,n) = l(l+1)/a^2, wavenumber(m,n) = l = m+n  (spherical.F90 convention)
  do n = 0, num_spherical
    do m = 0, num_fourier
      sw = real(m + n)
      eigen(m,n)   = sw*(sw + 1.0)/(radius*radius)
      wavenum(m,n) = m + n
    end do
  end do

  ! press_and_geopot stores pk/bk/vert_option in module state that implicit's
  ! pressure_variables calls read; init it with the same coefficients first.
  call press_and_geopot_init(pk, bk, .false., vopt)
  call implicit_init(pk, bk, ref_t, ref_surf_p, num_total, eigen, wavenum, alpha, vopt)

  ! random complex fields
  allocate(wr(nm,nn,K,ntime), wi(nm,nn,K,ntime))
  allocate(divs(nm,nn,K,ntime), ts(nm,nn,K,ntime), ln_ps(nm,nn,ntime))
  allocate(dt_divs(nm,nn,K), dt_ts(nm,nn,K), dt_ln_ps(nm,nn))

  call random_number(wr); call random_number(wi)
  divs = cmplx(wr - 0.5, wi - 0.5, kind=kind(1.0))
  call random_number(wr); call random_number(wi)
  ts = cmplx(220.0*(wr-0.5)*0.01, (wi-0.5)*0.01, kind=kind(1.0))  ! small T perturbations
  call random_number(wr(:,:,1,:)); call random_number(wi(:,:,1,:))
  ln_ps = cmplx(0.01*(wr(:,:,1,:)-0.5), 0.01*(wi(:,:,1,:)-0.5), kind=kind(1.0))
  call random_number(wr(:,:,:,1)); call random_number(wi(:,:,:,1))
  dt_divs = cmplx(wr(:,:,:,1)-0.5, wi(:,:,:,1)-0.5, kind=kind(1.0))
  call random_number(wr(:,:,:,1)); call random_number(wi(:,:,:,1))
  dt_ts = cmplx((wr(:,:,:,1)-0.5)*0.001, (wi(:,:,:,1)-0.5)*0.001, kind=kind(1.0))
  call random_number(wr(:,:,1,1)); call random_number(wi(:,:,1,1))
  dt_ln_ps = cmplx((wr(:,:,1,1)-0.5)*1.0e-6, (wi(:,:,1,1)-0.5)*1.0e-6, kind=kind(1.0))

  ! scalar metadata + coefficient arrays
  call jsca_dump_1d('imp_'//tag//'_meta', &
       (/ real(num_fourier), real(num_spherical), real(K), real(ntime), &
          real(num_total), alpha, dt, ref_surf_p, real(prev), real(curr) /))
  call jsca_dump_1d('imp_'//tag//'_pk', pk)
  call jsca_dump_1d('imp_'//tag//'_bk', bk)
  call jsca_dump_1d('imp_'//tag//'_ref_t', ref_t)
  call jsca_dump_2d('imp_'//tag//'_eigen', eigen)
  call jsca_dump_2d('imp_'//tag//'_wavenum', real(wavenum))

  ! inputs
  call dump4('imp_'//tag//'_divs', divs)
  call dump4('imp_'//tag//'_ts', ts)
  call dump3('imp_'//tag//'_ln_ps', ln_ps)
  call dump3('imp_'//tag//'_dt_divs_in', dt_divs)
  call dump3('imp_'//tag//'_dt_ts_in', dt_ts)
  call dump2('imp_'//tag//'_dt_ln_ps_in', dt_ln_ps)

  call implicit_correction(dt_divs, dt_ts, dt_ln_ps, divs, ts, ln_ps, dt, prev, curr)

  ! outputs
  call dump3('imp_'//tag//'_dt_divs_out', dt_divs)
  call dump3('imp_'//tag//'_dt_ts_out', dt_ts)
  call dump2('imp_'//tag//'_dt_ln_ps_out', dt_ln_ps)

  call implicit_end()
  call press_and_geopot_end()
  deallocate(pk, bk, ref_t, eigen, wavenum, wr, wi, divs, ts, ln_ps, dt_divs, dt_ts, dt_ln_ps)
end subroutine run_one

subroutine dump2(name, z)
  character(len=*), intent(in) :: name
  complex, intent(in) :: z(:,:)
  call jsca_dump_2d(name//'_re', real(z)); call jsca_dump_2d(name//'_im', aimag(z))
end subroutine dump2
subroutine dump3(name, z)
  character(len=*), intent(in) :: name
  complex, intent(in) :: z(:,:,:)
  call jsca_dump_3d(name//'_re', real(z)); call jsca_dump_3d(name//'_im', aimag(z))
end subroutine dump3
subroutine dump4(name, z)
  character(len=*), intent(in) :: name
  complex, intent(in) :: z(:,:,:,:)
  call jsca_dump_4d(name//'_re', real(z)); call jsca_dump_4d(name//'_im', aimag(z))
end subroutine dump4

end program dump_implicit_reference
