!! Golden-fixture driver for Isca's dry_convection scheme (dry_convection.f90),
!! the Schneider-Walker dry convective adjustment (convection_scheme='DRY').
!!
!! Compiles the REAL, unmodified dry_convection.f90 against fms_stubs.F90
!! (fms_mod + constants_mod) and rad_stubs.F90 (time_manager_mod + diag_manager_mod;
!! register_diag_field returns 0 so send_data is never called). Namelist
!! (tau, gamma) is read from a real input.nml via the stub open_namelist_file.
!!
!! Builds columns whose static stability sweeps from sub-adiabatic (stable, no
!! convection) to super-adiabatic (unstable, deep adjustment), so CAPE/CIN, the
!! LCL/LZB search and the energy-conserving adjustment are all exercised.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 rad_stubs.F90 jsca_dump.F90 \
!!     $ISCA_SRC/src/atmos_param/dry_convection/dry_convection.f90 \
!!     dump_dry_convection_reference.F90 -o dump_dry_convection_reference
!! Run with JSCA_DUMP_DIR set, then convert with read_dumps.py.

program dump_dry_convection_reference
  use dry_convection_mod, only: dry_convection_init, dry_convection
  use     time_manager_mod, only: time_type
  use          jsca_dump_mod, only: jsca_dump_2d, jsca_dump_3d

  implicit none

  integer, parameter :: ni = 3, nj = 6, nz = 25
  integer :: axes(4) = (/1, 2, 3, 4/)
  type(time_type) :: Time
  real, dimension(ni,nj,nz)   :: t, p_full, dt_tg
  real, dimension(ni,nj,nz+1) :: p_half
  real, dimension(ni,nj)      :: cape, cin
  integer, dimension(ni,nj)   :: lzb, lcl
  real :: sig_e(nz+1), sig_f(nz), ps, theta0, kappa, f
  integer :: i, j, k, unit

  Time = time_type(0, 0)

  ! --- dry_convection namelist (tau, gamma) ---
  open(newunit=unit, file='input.nml', status='replace')
  write(unit,'(a)') '&dry_convection_nml tau=21600.0, gamma=1.0 /'
  close(unit)

  call dry_convection_init(axes, Time)

  ps = 1.0e5
  kappa = 287.04 / 1004.64        ! Rd/cp
  theta0 = 300.0

  do k = 1, nz+1
    sig_e(k) = real(k-1) / real(nz)
  end do
  do k = 1, nz
    sig_f(k) = 0.5 * (sig_e(k) + sig_e(k+1))
  end do

  do j = 1, nj
    ! static-stability exponent: f=1 is neutral (dry adiabat); f<1 stable,
    ! f>1 super-adiabatic (convecting). Sweep -0.15 .. +0.35 across nj.
    f = 0.85 + 0.10 * real(j-1)
    do i = 1, ni
      do k = 1, nz
        p_full(i,j,k) = sig_f(k) * ps
        p_half(i,j,k) = sig_e(k) * ps
        ! t = theta0 * (p/ps)**(kappa*f); a small lat-dependent warm bump near
        ! the middle so the LCL/LZB land in the interior for some columns.
        t(i,j,k) = theta0 * (p_full(i,j,k) / ps) ** (kappa * f) &
                   + 2.0 * real(i-1) * sig_f(k) * (1.0 - sig_f(k))
        t(i,j,k) = max(t(i,j,k), 150.0)
      end do
      p_half(i,j,nz+1) = ps
    end do
  end do

  ! Last lat row: an ELEVATED unstable layer (stable/inversion near the surface,
  ! super-adiabatic aloft) so the LCL is raised off the ground (exercises the
  ! lcl-setting branch, which the surface-based columns above leave at btm).
  do i = 1, ni
    ! Relative to the surface parcel's dry adiabat (theta0*(p/ps)**kappa): a warm
    ! stable cap near the surface (parcel colder -> CIN), a cold unstable layer in
    ! the middle (parcel warmer -> CAPE; LCL set at the base of it), then a strongly
    ! stable layer aloft (LZB in the interior).
    do k = 1, nz
      t(i,nj,k) = theta0 * (p_full(i,nj,k) / ps) ** kappa
      if (sig_f(k) > 0.75) then
        t(i,nj,k) = t(i,nj,k) + 4.0
      else if (sig_f(k) > 0.45) then
        t(i,nj,k) = t(i,nj,k) - 6.0
      else
        t(i,nj,k) = t(i,nj,k) + 20.0
      end if
    end do
  end do

  call dry_convection(Time, t, p_full, p_half, dt_tg, cape, cin, lzb, lcl)

  call jsca_dump_3d('dc_t',      t)
  call jsca_dump_3d('dc_pfull',  p_full)
  call jsca_dump_3d('dc_phalf',  p_half)
  call jsca_dump_3d('dc_dt_tg',  dt_tg)
  call jsca_dump_2d('dc_cape',   cape)
  call jsca_dump_2d('dc_cin',    cin)
  call jsca_dump_2d('dc_lzb',    real(lzb))
  call jsca_dump_2d('dc_lcl',    real(lcl))

  write(*,*) 'DRY_CONV_DUMP_DONE  dt_tg min/max:', minval(dt_tg), maxval(dt_tg)
  write(*,*) '  cape min/max:', minval(cape), maxval(cape)
  write(*,*) '  lzb min/max:', minval(lzb), maxval(lzb), ' lcl min/max:', minval(lcl), maxval(lcl)
end program dump_dry_convection_reference
