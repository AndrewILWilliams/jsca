!! Minimal stubs for fms_mod / constants_mod so numerical kernels from the
!! Isca tree (e.g. gauss_and_legendre.F90) compile standalone, without FMS.
!! Only logging/bookkeeping is stubbed — no numerics live here.

module constants_mod
implicit none
! Values must mirror Isca's constants.F90 (and jsca/constants.py) exactly.
real, public, parameter :: pi     = 3.14159265358979323846
real, public, parameter :: grav   = 9.80
real, public, parameter :: rdgas  = 287.04
real, public, parameter :: rvgas  = 461.50
real, public, parameter :: kappa  = 2./7.
real, public, parameter :: cp_air = rdgas/kappa   ! = EARTH_CP_AIR (constants.F90 L86)
real, public, parameter :: radius = 6376.0e3      ! Isca default RADIUS (constants.F90 L254)
real, public, parameter :: seconds_per_day = 8.640000e4  ! constants.F90 L176
real, public, parameter :: stefan  = 5.6734e-8    ! constants.F90 L238
real, public, parameter :: pstd_mks_earth = 101325.0    ! constants.F90 L252
real, public :: pstd_mks = 101325.0               ! constants.F90 L263 (= earth default)
real, public :: seconds_per_sol = 8.640000e4      ! Earth default (do_seasonal path only)
real, public, parameter :: hlv     = 2.500e6      ! constants.F90 L123
real, public, parameter :: hlf     = 3.34e5       ! constants.F90 L124
real, public, parameter :: hls     = hlv + hlf    ! constants.F90 L125
real, public, parameter :: tfreeze = 273.16       ! constants.F90 L126
real, public, parameter :: es0     = 1.0          ! constants.F90 DEF_ES0 (L119)
real, public :: orbital_period = 365.25*8.640000e4  ! EARTH_ORBITAL_PERIOD (mutable; hs_forcing_nml member)
real, public :: solar_const = 1368.22             ! constants.F90 L260
end module constants_mod

module fms_mod
implicit none
integer, public, parameter :: FATAL = 2
integer, public, parameter :: NOTE = 0

contains

integer function mpp_pe()
  mpp_pe = 0
end function mpp_pe

integer function mpp_root_pe()
  mpp_root_pe = 0
end function mpp_root_pe

integer function mpp_npes()
  mpp_npes = 1
end function mpp_npes

subroutine error_mesg(routine, message, level)
  character(len=*), intent(in) :: routine, message
  integer, intent(in) :: level
  write(*,*) 'ERROR (', trim(routine), '): ', trim(message)
  if (level == FATAL) stop 1
end subroutine error_mesg

subroutine mpp_error(level, message)
  integer, intent(in) :: level
  character(len=*), intent(in), optional :: message
  if (present(message)) write(*,*) 'MPP_ERROR: ', trim(message)
  if (level == FATAL) stop 1
end subroutine mpp_error

! Returns .true. (and sets err_msg) only when an error message is present, so a
! caller that passes err_msg gets a clean return; mirrors FMS's helper contract.
logical function fms_error_handler(routine, message, err_msg)
  character(len=*), intent(in) :: routine, message
  character(len=*), intent(out), optional :: err_msg
  if (present(err_msg)) then
    err_msg = message
    fms_error_handler = .true.
  else
    write(*,*) 'ERROR (', trim(routine), '): ', trim(message)
    fms_error_handler = .false.
  endif
end function fms_error_handler

subroutine write_version_number(version, tagname)
  character(len=*), intent(in) :: version, tagname
end subroutine write_version_number

integer function stdout()
  stdout = 6
end function stdout

integer function stdlog()
  stdlog = 6
end function stdlog

integer function open_namelist_file()
  open_namelist_file = 10
end function open_namelist_file

! Real file open (I/O plumbing, no numerics). Some modules (e.g.
! qe_moist_convection) read their namelist from a real input.nml via open_file
! and also append to logfile.out; the fixture driver writes that input.nml. A
! newunit-based open keeps this independent of the fixed unit from
! open_namelist_file. action='read'/'write'/'append' map to the obvious opens.
integer function open_file(file, action, form, threading, recl)
  character(len=*), intent(in)           :: file
  character(len=*), intent(in), optional :: action, form, threading
  integer,          intent(in), optional :: recl
  integer :: unit
  character(len=16) :: act
  act = 'read'
  if (present(action)) act = action
  if (trim(act) == 'append') then
    open(newunit=unit, file=trim(file), status='unknown', position='append')
  else if (trim(act) == 'write') then
    open(newunit=unit, file=trim(file), status='replace')
  else
    open(newunit=unit, file=trim(file), status='old', action='read')
  end if
  open_file = unit
end function open_file

subroutine close_file(unit, status)
  integer, intent(in) :: unit
  character(len=*), intent(in), optional :: status
  logical :: is_open
  inquire(unit=unit, opened=is_open)
  if (is_open) close(unit)
end subroutine close_file

integer function check_nml_error(io, name)
  integer, intent(in) :: io
  character(len=*), intent(in) :: name
  check_nml_error = 0
end function check_nml_error

! Real existence check via INQUIRE. In sandboxes with no input.nml this returns
! .false. exactly as before (modules fall back to namelist defaults); a driver
! that writes an input.nml (e.g. the qe_moist_convection fixture) gets .true.
logical function file_exist(name)
  character(len=*), intent(in) :: name
  inquire(file=trim(name), exist=file_exist)
end function file_exist

function uppercase(cs) result(ucs)
  character(len=*), intent(in) :: cs
  character(len=len(cs)) :: ucs
  integer :: k, ia
  ucs = cs
  do k = 1, len_trim(cs)
    ia = iachar(cs(k:k))
    if (ia >= iachar('a') .and. ia <= iachar('z')) ucs(k:k) = achar(ia - 32)
  end do
end function uppercase

subroutine set_domain(domain)
  integer, intent(in) :: domain
end subroutine set_domain

subroutine read_data(filename, fieldname, data, domain)
  character(len=*), intent(in) :: filename, fieldname
  real, intent(inout) :: data(:,:)
  integer, intent(in) :: domain
end subroutine read_data

subroutine write_data(filename, fieldname, data, domain)
  character(len=*), intent(in) :: filename, fieldname
  real, intent(in) :: data(:,:)
  integer, intent(in) :: domain
end subroutine write_data

end module fms_mod
