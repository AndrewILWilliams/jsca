!! Golden-fixture driver for Isca's simplified Betts-Miller (Frierson
!! quasi-equilibrium) convection, qe_moist_convection.F90, with the Frierson
!! namelist (qe_moist_convection_nml: rhbm=0.7, Tmin=160, Tmax=350; tau_bm
!! default 7200).
!!
!! Compiles the REAL, unmodified qe_moist_convection.F90 and the real
!! sat_vapor_pres.F90 wrapper (+ kernel) against the shared stubs. qe reads its
!! namelist from a real input.nml (this driver writes it) through the stub
!! open_file; the sat_vapor_pres wrapper's do_simple=.true. comes through FMS's
!! internal-file buffer (build -DINTERNAL_FILE_NML). Both coexist.
!!
!! The column set spans the scheme's regimes: warm/moist soundings that convect
!! (deep and shallow) and cool/dry soundings with no CAPE. Dumps the full public
!! output (deltaT, deltaq, rain, CAPE, CIN, Tref, qref, convflag, kLZBs, kLCLs)
!! for tests/test_qe_moist_convection_fixtures.py.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     -DINTERNAL_FILE_NML \
!!     fms_stubs.F90 mpp_mod_stub.F90 mpp_io_stub.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres_k.F90 \
!!     $ISCA_SRC/src/shared/sat_vapor_pres/sat_vapor_pres.F90 \
!!     $ISCA_SRC/src/atmos_param/qe_moist_convection/qe_moist_convection.F90 \
!!     dump_qe_moist_convection_reference.F90 -o dump_qe_moist_convection_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_qe_moist_convection_reference
  use                 mpp_mod, only: input_nml_file
  use qe_moist_convection_mod, only: qe_moist_convection, qe_moist_convection_init
  use            jsca_dump_mod, only: jsca_dump_2d, jsca_dump_3d

  implicit none

  integer, parameter :: ni = 6, nj = 5, kx = 25
  real, dimension(ni,nj,kx)   :: tin, qin, pfull, deltaT, deltaq, qref, tref
  real, dimension(ni,nj,kx+1) :: phalf
  real, dimension(ni,nj)      :: rain, snow, cape, cin, invtau_q, invtau_t
  real, dimension(ni,nj)      :: convflag_r, klzb_r, klcl_r, ps
  integer, dimension(ni,nj)   :: convflag, klzbs, klcls
  logical, dimension(ni,nj)   :: coldT
  real    :: sig_e(kx+1), sig_f(kx), dt, tsurf, rh0, rhtop, rh, gamma, lat, frac
  integer :: i, j, k, unit

  ! --- namelists: sat_vapor_pres via internal buffer, qe via a real input.nml ---
  allocate(input_nml_file(2))
  input_nml_file(1) = '&sat_vapor_pres_nml do_simple=.true. /'
  input_nml_file(2) = ' '
  open(newunit=unit, file='input.nml', status='replace')
  write(unit,'(a)') '&qe_moist_convection_nml'
  write(unit,'(a)') '  rhbm = 0.7, Tmin = 160.0, Tmax = 350.0'
  write(unit,'(a)') '/'
  close(unit)

  call qe_moist_convection_init()

  dt = 720.0
  coldT = .false.

  ! sigma half-levels (top=0 -> surface=1), full-levels at midpoints
  do k = 1, kx+1
    sig_e(k) = real(k-1) / real(kx)
  end do
  do k = 1, kx
    sig_f(k) = 0.5 * (sig_e(k) + sig_e(k+1))
  end do

  do j = 1, nj
    lat  = -30.0 + 60.0 * real(j-1) / real(nj-1)          ! -30 .. +30 deg (tropics)
    do i = 1, ni
      frac = real(i-1) / real(ni-1)                       ! 0 .. 1 across columns
      ps(i,j) = 1.0e5
      ! surface temperature: warm & moist (convecting) -> cool & dry (stable)
      tsurf = 302.0 - 14.0 * frac - 6.0 * (sin(lat*3.14159/180.0))**2
      ! Conditionally unstable lapse rate (kept away from neutral so deep columns
      ! have Pt clearly greater than Pq -> the Tref-shift branch, avoiding the
      ! Pq==Pt knife-edge). Spans deep / shallow / no-CAPE across the grid.
      gamma = 6.8e-3 + 0.5e-3 * real(j-1) + 1.2e-3 * frac  ! lapse rate (K/m)
      rh0   = 0.90 - 0.55 * frac                          ! low-level RH
      rhtop = 0.20

      do k = 1, kx
        phalf(i,j,k) = sig_e(k) * ps(i,j)
        ! temperature via a hydrostatic-ish lapse in log-pressure, capped at a
        ! stratospheric floor so upper levels are stable
        tin(i,j,k) = tsurf - (gamma * 7000.0) * (-log(max(sig_f(k), 1.0e-3)))
        tin(i,j,k) = max(tin(i,j,k), 195.0)
      end do
      phalf(i,j,kx+1) = ps(i,j)
      do k = 1, kx
        pfull(i,j,k) = 0.5 * (phalf(i,j,k) + phalf(i,j,k+1))
      end do
    end do
  end do
  where (pfull < 100.0) pfull = 100.0

  ! humidity: RH decreasing with height, converted to specific humidity with a
  ! simple Clausius-Clapeyron es (do_simple form) so inputs are self-consistent.
  do j = 1, nj
    do i = 1, ni
      frac = real(i-1) / real(ni-1)
      rh0   = 0.97 - 0.62 * frac                         ! near-saturated -> dry across columns
      rhtop = 0.60 - 0.45 * frac                          ! keep upper levels moist too (wettest cols)
      do k = 1, kx
        ! High RH sustained through much of the column for the moistest columns
        ! maximises the moisture excess over the rhbm=0.7 reference (Pq), so some
        ! reach deep convection with Pq>Pt (the time-scale-change branch); drier
        ! columns stay Tref-shift deep / shallow / no-CAPE.
        rh = rhtop + (rh0 - rhtop) * sig_f(k)**(0.4 + 0.3 * frac)
        qin(i,j,k) = rh * qsat_simple(tin(i,j,k), pfull(i,j,k))
      end do
    end do
  end do

  call qe_moist_convection(dt, tin, qin, pfull, phalf, coldT, rain, snow, &
       deltaT, deltaq, qref, convflag, klzbs, cape, cin, invtau_q, invtau_t, &
       tref, klcls)

  convflag_r = real(convflag)
  klzb_r     = real(klzbs)
  klcl_r     = real(klcls)

  call jsca_dump_3d('qe_tin',    tin)
  call jsca_dump_3d('qe_qin',    qin)
  call jsca_dump_3d('qe_pfull',  pfull)
  call jsca_dump_3d('qe_phalf',  phalf)
  call jsca_dump_3d('qe_deltaT', deltaT)
  call jsca_dump_3d('qe_deltaq', deltaq)
  call jsca_dump_3d('qe_tref',   tref)
  call jsca_dump_3d('qe_qref',   qref)
  call jsca_dump_2d('qe_rain',   rain)
  call jsca_dump_2d('qe_cape',   cape)
  call jsca_dump_2d('qe_cin',    cin)
  call jsca_dump_2d('qe_invtau_q', invtau_q)
  call jsca_dump_2d('qe_invtau_t', invtau_t)
  call jsca_dump_2d('qe_convflag', convflag_r)
  call jsca_dump_2d('qe_klzb',   klzb_r)
  call jsca_dump_2d('qe_klcl',   klcl_r)

  write(*,*) 'QE_DUMP_DONE  convflag counts (0/1/2):', &
       count(convflag==0), count(convflag==1), count(convflag==2)
  write(*,*) '  rain min/max:', minval(rain), maxval(rain)
  write(*,*) '  cape max:', maxval(cape)

contains

  ! do_simple saturation specific humidity, only for building self-consistent
  ! INPUT humidity (not part of the tested scheme). Mirrors sat_vapor_pres do_simple.
  real function qsat_simple(t, p)
    real, intent(in) :: t, p
    real :: es, eps
    eps = 287.04 / 461.50
    es  = 610.78 * exp(-2.5e6 / 461.50 * (1.0/t - 1.0/273.16))
    qsat_simple = eps * es / (p - (1.0 - eps) * es)
  end function qsat_simple

end program dump_qe_moist_convection_reference
