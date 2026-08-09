!! Golden-fixture driver for Isca's Byrne & O'Gorman (2013) grey radiation,
!! two_stream_gray_rad.F90, with rad_scheme='byrne' (do_seasonal=.false.).
!!
!! Identical harness to dump_two_stream_gray_rad_reference.F90 (the Frierson
!! fixture) but selects the Byrne longwave scheme. The Byrne LW optical depth is
!! humidity-dependent, so this driver builds a physically decreasing-with-height
!! specific-humidity profile q (rather than the constant q the Frierson driver
!! uses, since Frierson ignores q) to exercise the bog_b*q term.
!!
!! Compiles the REAL, unmodified two_stream_gray_rad.F90 against fms_stubs.F90
!! (fms_mod/constants_mod) + rad_stubs.F90 (diag/time/astronomy/interpolator --
!! all no-op on the non-seasonal path).
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 rad_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_param/two_stream_gray_rad/two_stream_gray_rad.F90 \
!!     dump_two_stream_gray_rad_byrne_reference.F90 \
!!     -o dump_two_stream_gray_rad_byrne_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_two_stream_gray_rad_byrne_reference
  use two_stream_gray_rad_mod, only: two_stream_gray_rad_init, &
                                     two_stream_gray_rad_down, two_stream_gray_rad_up
  use          time_manager_mod, only: time_type
  use               jsca_dump_mod, only: jsca_dump_2d, jsca_dump_3d

  implicit none

  integer, parameter :: ni = 3, nj = 8, nz = 25
  integer :: axes(4) = (/1, 2, 3, 4/)
  type(time_type) :: Time
  real, dimension(ni,nj,nz)   :: t, q, tdt
  real, dimension(ni,nj,nz+1) :: p_half
  real, dimension(ni,nj)      :: lat, lon, t_surf, albedo, net_surf_sw_down, surf_lw_down
  real :: sig_e(nz+1), sig_f(nz), ps, latdeg, pi, qsurf
  integer :: i, j, k, unit

  pi = 4.0 * atan(1.0)
  Time = time_type(0, 0)

  ! --- Byrne radiation namelist (rad_scheme='byrne') ---
  open(newunit=unit, file='input.nml', status='replace')
  write(unit,'(a)') '&two_stream_gray_rad_nml'
  write(unit,'(a)') "  rad_scheme = 'byrne', do_seasonal = .false., atm_abs = 0.2"
  write(unit,'(a)') '/'
  close(unit)

  call two_stream_gray_rad_init(1, ni, 1, nj, nz, axes, Time, dt_real=720.0)

  ! sigma half-levels (top=0 -> surface=1), full-levels at midpoints
  do k = 1, nz+1
    sig_e(k) = real(k-1) / real(nz)
  end do
  do k = 1, nz
    sig_f(k) = 0.5 * (sig_e(k) + sig_e(k+1))
  end do

  ps = 1.0e5
  albedo = 0.31        ! surface albedo

  do j = 1, nj
    latdeg = -84.0 + 168.0 * real(j-1) / real(nj-1)     ! -84 .. +84 deg
    do i = 1, ni
      lat(i,j) = latdeg * pi / 180.0
      lon(i,j) = 2.0 * pi * real(i-1) / real(ni)        ! (unused)
      ! warm equator, cold poles at the surface
      t_surf(i,j) = 290.0 - 40.0 * sin(lat(i,j))**2
      ! surface humidity: moist tropics, dry poles (kg/kg)
      qsurf = 0.018 * (1.0 - sin(lat(i,j))**2) + 1.0e-4
      do k = 1, nz
        p_half(i,j,k) = sig_e(k) * ps
        ! temperature: ~stratosphere aloft, warm near surface, lat-dependent
        t(i,j,k) = (285.0 - 35.0 * sin(lat(i,j))**2) - 70.0 * (1.0 - sig_f(k))
        t(i,j,k) = max(t(i,j,k), 190.0)
        ! specific humidity decreasing with height (roughly q ~ qsurf * sigma^3),
        ! floored: exercises the humidity-dependent Byrne LW optical depth
        q(i,j,k) = max(qsurf * sig_f(k)**3, 1.0e-6)
      end do
      p_half(i,j,nz+1) = ps
    end do
  end do

  net_surf_sw_down = 0.0
  surf_lw_down = 0.0
  tdt = 0.0

  call two_stream_gray_rad_down(1, 1, Time, lat, lon, p_half, t, &
                                net_surf_sw_down, surf_lw_down, albedo, q)
  call two_stream_gray_rad_up(1, 1, Time, lat, p_half, t_surf, t, tdt, albedo)

  call jsca_dump_2d('rad_lat',    lat)
  call jsca_dump_3d('rad_phalf',  p_half)
  call jsca_dump_3d('rad_t',      t)
  call jsca_dump_3d('rad_q',      q)
  call jsca_dump_2d('rad_tsurf',  t_surf)
  call jsca_dump_2d('rad_albedo', albedo)
  call jsca_dump_3d('rad_tdt',    tdt)                  ! = tdt_rad (tdt passed as 0)
  call jsca_dump_2d('rad_net_sw_sfc', net_surf_sw_down)
  call jsca_dump_2d('rad_lw_down_sfc', surf_lw_down)

  write(*,*) 'RAD_BYRNE_DUMP_DONE  tdt min/max:', minval(tdt), maxval(tdt)
  write(*,*) '  net_sw_sfc min/max:', minval(net_surf_sw_down), maxval(net_surf_sw_down)
  write(*,*) '  lw_down_sfc min/max:', minval(surf_lw_down), maxval(surf_lw_down)
end program dump_two_stream_gray_rad_byrne_reference
