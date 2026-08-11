!! Golden-fixture driver for Isca's full Betts-Miller convection (betts_miller.f90),
!! convection_scheme='FULL_BETTS_MILLER', default namelist (do_simp=.true.,
!! do_shallower/do_changeqref/do_envsat/do_taucape=.false., buoyancy_kick=0).
!!
!! Compiles the REAL, unmodified betts_miller.f90 with the REAL sat_vapor_pres
!! wrapper + kernel (escomp/descomp) against the shared stubs. sat_vapor_pres runs
!! with do_simple=.true. (the column/Frierson setting), so escomp is the simple
!! closed form. Builds a column set that sweeps convecting (warm/moist, deep CAPE)
!! to non-convecting (dry/stable), exercising capecalcnew, the reference-profile
!! relaxation, and the do_simp energy conservation.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -cpp \
!!     -DINTERNAL_FILE_NML -fallow-argument-mismatch \
!!     fms_stubs.F90 mpp_mod_stub.F90 mpp_io_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres_k.F90 \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres.F90 \
!!     $ISCA_SRC/src/atmos_param/betts_miller/betts_miller.f90 \
!!     dump_betts_miller_reference.F90 -o dump_betts_miller_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_betts_miller_reference
  use     betts_miller_mod, only: betts_miller, betts_miller_init
  use sat_vapor_pres_mod, only: sat_vapor_pres_init
  use          mpp_mod, only: input_nml_file
  use       jsca_dump_mod, only: jsca_dump_2d, jsca_dump_3d

  implicit none

  integer, parameter :: ni = 3, nj = 6, nz = 25
  real, dimension(ni,nj,nz)   :: t, q, pfull, tdel, qdel, q_ref, t_ref
  real, dimension(ni,nj,nz+1) :: phalf
  real, dimension(ni,nj)      :: rain, snow, cape, cin, invtau_bm_t, invtau_bm_q, capeflag
  integer, dimension(ni,nj)   :: bmflag, klzbs, klcls
  logical, dimension(ni,nj)   :: coldT
  real :: sig_e(nz+1), sig_f(nz), ps, tsurf, lapse, qsurf, dt
  integer :: i, j, k, unit

  ! --- namelists: sat_vapor_pres do_simple=.true.; betts_miller defaults ---
  allocate(input_nml_file(3))
  input_nml_file(1) = '&sat_vapor_pres_nml do_simple=.true. /'
  input_nml_file(2) = '&betts_miller_nml tau_bm=7200.0, rhbm=0.8 /'
  input_nml_file(3) = ' '
  open(newunit=unit, file='input.nml', status='replace')
  write(unit,'(a)') '&sat_vapor_pres_nml do_simple=.true. /'
  write(unit,'(a)') '&betts_miller_nml tau_bm=7200.0, rhbm=0.8 /'
  close(unit)

  call sat_vapor_pres_init
  call betts_miller_init()

  dt = 600.0
  ps = 1.0e5
  coldT = .false.

  do k = 1, nz+1
    sig_e(k) = real(k-1) / real(nz)
  end do
  do k = 1, nz
    sig_f(k) = 0.5 * (sig_e(k) + sig_e(k+1))
  end do

  do j = 1, nj
    ! surface temperature / humidity sweep: warm+moist (deep convection) to
    ! cool+dry (no CAPE). j=1 warmest/moistest, j=nj coolest/driest.
    tsurf = 302.0 - 6.0 * real(j-1)
    qsurf = 0.020 - 0.0032 * real(j-1)
    lapse = 7.5e-3
    do i = 1, ni
      do k = 1, nz
        pfull(i,j,k) = sig_f(k) * ps
        phalf(i,j,k) = sig_e(k) * ps
        ! temperature: warm surface, ~7.5 K/km lapse (conditionally unstable),
        ! isothermal stratosphere; small lon-dependent tweak for variety.
        t(i,j,k) = tsurf - lapse * 7000.0 * (1.0 - sig_f(k)) + 0.5 * real(i-1)
        t(i,j,k) = max(t(i,j,k), 195.0)
        ! humidity: moist near surface decaying upward
        q(i,j,k) = max(qsurf * sig_f(k)**3, 1.0e-6)
      end do
      phalf(i,j,nz+1) = ps
    end do
  end do

  call betts_miller(dt, t, q, pfull, phalf, coldT, rain, snow, tdel, qdel, &
                    q_ref, bmflag, klzbs, cape, cin, t_ref, invtau_bm_t, &
                    invtau_bm_q, capeflag, klcls)

  call jsca_dump_3d('bm_t',      t)
  call jsca_dump_3d('bm_q',      q)
  call jsca_dump_3d('bm_pfull',  pfull)
  call jsca_dump_3d('bm_phalf',  phalf)
  call jsca_dump_3d('bm_tdel',   tdel)
  call jsca_dump_3d('bm_qdel',   qdel)
  call jsca_dump_2d('bm_rain',   rain)
  call jsca_dump_2d('bm_cape',   cape)
  call jsca_dump_2d('bm_cin',    cin)
  call jsca_dump_2d('bm_klzb',   real(klzbs))
  call jsca_dump_2d('bm_klcl',   real(klcls))
  call jsca_dump_2d('bm_bmflag', real(bmflag))

  write(*,*) 'BM_DUMP_DONE  rain min/max:', minval(rain), maxval(rain)
  write(*,*) '  cape min/max:', minval(cape), maxval(cape)
  write(*,*) '  klzb min/max:', minval(klzbs), maxval(klzbs), ' bmflag:', minval(bmflag), maxval(bmflag)
end program dump_betts_miller_reference
