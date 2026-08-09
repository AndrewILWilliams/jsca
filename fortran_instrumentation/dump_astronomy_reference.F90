!! Golden-fixture driver for Isca's astronomy module (diurnal_solar), the
!! insolation engine behind two_stream_gray_rad's do_seasonal path.
!!
!! Compiles the REAL, unmodified astronomy.f90 against fms_stubs.F90
!! (fms_mod + constants_mod) and astronomy_stubs.F90 (time_manager_mod + mpp_mod).
!! The diurnal_solar path never calls the time_manager routines (it is handed
!! gmt / time_since_ae directly), so the stubs are compile-only.
!!
!! Calls astronomy_init (default orbital params: ecc=0, obliq=23.439 deg,
!! per=102.932 deg, num_angles=3600), then diurnal_solar over a lat/lon grid for
!! a sweep of orbital positions (time_since_ae) and gmt, both instantaneous and
!! time-averaged over dt. Dumps cosz, fracday and, per case, a 4-vector
!! [gmt, time_since_ae, dt, rrsun] (dt<0 flags the instantaneous case).
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 astronomy_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/shared/astronomy/astronomy.f90 \
!!     dump_astronomy_reference.F90 -o dump_astronomy_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_astronomy_reference
  use astronomy_mod, only: astronomy_init, diurnal_solar
  use     jsca_dump_mod, only: jsca_dump_1d, jsca_dump_2d

  implicit none

  integer, parameter :: nlat = 17, nlon = 8
  real, dimension(nlat,nlon) :: lat, lon, cosz, fracday
  real :: rrsun, pi, twopi, gmt, tsae, dt, meta(4)
  real :: times(4), gmts(2)
  integer :: i, j, it, ig, ia

  pi = 4.0 * atan(1.0)
  twopi = 2.0 * pi

  ! lat -80..80 deg, lon 0..(nlon-1)/nlon * 2pi
  do j = 1, nlon
    do i = 1, nlat
      lat(i,j) = (-80.0 + 160.0 * real(i-1) / real(nlat-1)) * pi / 180.0
      lon(i,j) = twopi * real(j-1) / real(nlon)
    end do
  end do

  ! orbital positions: autumn equinox, and +1/4, +1/2, +3/4 year (radians)
  times = (/ 0.0, 0.5*pi, pi, 1.5*pi /)
  gmts  = (/ 0.0, pi /)                 ! two times of day

  call astronomy_init

  call jsca_dump_2d('astr_lat', lat)
  call jsca_dump_2d('astr_lon', lon)

  do it = 1, 4
    tsae = times(it)
    ! nudge exact 0 upward: astronomy asserts 0 <= time_since_ae <= twopi
    if (tsae <= 0.0) tsae = 1.0e-6
    do ig = 1, 2
      gmt = gmts(ig)
      do ia = 1, 2
        if (ia == 1) then
          ! instantaneous
          call diurnal_solar(lat, lon, gmt, tsae, cosz, fracday, rrsun)
          dt = -1.0
        else
          ! time-averaged over dt = 3 hours (in radians of a day)
          dt = (3.0 * 3600.0 / 86400.0) * twopi
          call diurnal_solar(lat, lon, gmt, tsae, cosz, fracday, rrsun, dt)
        end if
        meta = (/ gmt, tsae, dt, rrsun /)
        call jsca_dump_1d('astr_meta',    meta)
        call jsca_dump_2d('astr_cosz',    cosz)
        call jsca_dump_2d('astr_fracday', fracday)
      end do
    end do
  end do

  write(*,*) 'ASTRONOMY_DUMP_DONE  16 cases (4 orbital x 2 gmt x inst/avg)'
end program dump_astronomy_reference
