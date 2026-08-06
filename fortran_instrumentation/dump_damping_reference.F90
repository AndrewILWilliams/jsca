!! Golden-fixture driver for Isca's top-of-model Rayleigh sponge
!! (damping_driver.f90 -> rayleigh), Frierson settings (damping_driver_nml:
!! do_rayleigh=.true., trayfric=-0.25, sponge_pbottom=5000., do_conserve_energy=
!! .true.; all gravity-wave-drag paths .false.).
!!
!! damping_driver_init reads the namelist and, from the reference full-level
!! pressures pref (built here from the Frierson pure-sigma bk with ps=PSTD_MKS),
!! sets nlev_rayfric = the level closest to 2*sponge_pbottom. damping_driver then
!! calls rayleigh, which relaxes u,v toward zero in the top nlev_rayfric levels
!! where p_full < sponge_pbottom, with the conserved KE reappearing as heating.
!!
!! udt/vdt/tdt are intent(inout) accumulators; we zero them before the call so the
!! dump is the pure sponge tendency. The 3-D p_full is the reference profile
!! scaled per column so the sponge's pressure threshold falls at different levels
!! across the grid (exercises both the level-count bound and the where-clause).
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 damping_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_param/damping_driver/damping_driver.f90 \
!!     dump_damping_reference.F90 -o dump_damping_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_damping_reference
  use damping_driver_mod, only: damping_driver_init, damping_driver
  use   time_manager_mod, only: time_type
  use      constants_mod, only: pstd_mks
  use      jsca_dump_mod, only: jsca_dump_1d, jsca_dump_3d

  implicit none

  integer, parameter :: nlon = 4, nlat = 6, nz = 25
  ! Frierson pure-sigma vertical coordinate (frierson_test_case.py: pk=0, bk given)
  real, parameter :: bk(nz+1) = (/ &
       0.000000, 0.0117665, 0.0196679, 0.0315244, 0.0485411, 0.0719344, &
       0.1027829, 0.1418581, 0.1894648, 0.2453219, 0.3085103, 0.3775033, &
       0.4502789, 0.5244989, 0.5977253, 0.6676441, 0.7322627, 0.7900587, &
       0.8400683, 0.8819111, 0.9157609, 0.9422770, 0.9625127, 0.9778177, &
       0.9897489, 1.0000000 /)

  integer :: axes(4) = (/1, 2, 3, 4/)
  type(time_type) :: Time
  real :: lonb(nlon+1), latb(nlat+1), pref(nz+1), p_half_ref(nz+1), pfull_ref(nz)
  real, dimension(nlon,nlat)      :: sgsmtn, z_pbl
  real, dimension(nlon,nlat)      :: lat2d
  real, dimension(nlon,nlat,nz)   :: pfull, zfull, zhalf3, u, v, t, q
  real, dimension(nlon,nlat,nz+1) :: phalf3e
  real, dimension(nlon,nlat,nz)   :: udt, vdt, tdt, qdt
  real, dimension(nlon,nlat,nz,1) :: r, rdt
  real    :: dt, frac, scal
  integer :: i, j, k, unit

  Time = time_type(0, 0)
  dt = 720.0

  ! reference pressures: pure sigma p_half = bk*ps; pref(full) = mean of edges.
  do k = 1, nz+1
    p_half_ref(k) = bk(k) * pstd_mks
  end do
  do k = 1, nz
    pfull_ref(k) = 0.5 * (p_half_ref(k) + p_half_ref(k+1))
  end do
  pref(1:nz) = pfull_ref
  pref(nz+1) = pstd_mks   ! Isca appends the reference surface pressure

  ! --- Frierson damping_driver namelist (real input.nml via open_namelist_file) ---
  open(newunit=unit, file='input.nml', status='replace')
  write(unit,'(a)') '&damping_driver_nml'
  write(unit,'(a)') '  do_rayleigh = .true., trayfric = -0.25,'
  write(unit,'(a)') '  sponge_pbottom = 5000.0, do_conserve_energy = .true.'
  write(unit,'(a)') '/'
  close(unit)

  lonb = 0.0; latb = 0.0; sgsmtn = 0.0; z_pbl = 0.0; lat2d = 0.0
  call damping_driver_init(lonb, latb, pref, axes, Time, sgsmtn)

  ! --- build the 3-D column state ---
  do j = 1, nlat
    do i = 1, nlon
      frac = real((i-1) + (j-1)*nlon) / real(nlon*nlat - 1)   ! 0..1 across grid
      scal = 0.7 + 0.4 * frac        ! per-column pressure scaling (shifts the threshold)
      do k = 1, nz
        pfull(i,j,k) = pfull_ref(k) * scal
        u(i,j,k)     = 20.0 + 40.0 * frac    ! m/s
        v(i,j,k)     = -10.0 + 20.0 * frac
        t(i,j,k)     = 240.0
        q(i,j,k)     = 0.0
      end do
      do k = 1, nz+1
        phalf3e(i,j,k) = p_half_ref(k) * scal
      end do
    end do
  end do
  zfull = 0.0; zhalf3 = 0.0; r = 0.0
  udt = 0.0; vdt = 0.0; tdt = 0.0; qdt = 0.0; rdt = 0.0

  call damping_driver(1, 1, lat2d, Time, dt, pfull, phalf3e, zfull, zhalf3, &
                      u, v, t, q, r, udt, vdt, tdt, qdt, rdt, z_pbl)

  call jsca_dump_1d('dd_pref',   pref(1:nz))
  call jsca_dump_3d('dd_pfull',  pfull)
  call jsca_dump_3d('dd_u',      u)
  call jsca_dump_3d('dd_v',      v)
  call jsca_dump_3d('dd_udt',    udt)
  call jsca_dump_3d('dd_vdt',    vdt)
  call jsca_dump_3d('dd_tdt',    tdt)

  write(*,*) 'DD_DUMP_DONE  udt range', minval(udt), maxval(udt), &
             '  tdt range', minval(tdt), maxval(tdt)
end program dump_damping_reference
