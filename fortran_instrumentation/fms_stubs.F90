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
real, public :: orbital_period = 365.25*8.640000e4  ! EARTH_ORBITAL_PERIOD (mutable; hs_forcing_nml member)
real, public :: solar_const = 1368.22             ! constants.F90 L260
end module constants_mod

module fms_mod
implicit none
integer, public, parameter :: FATAL = 2

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

subroutine close_file(unit)
  integer, intent(in) :: unit
end subroutine close_file

integer function check_nml_error(io, name)
  integer, intent(in) :: io
  character(len=*), intent(in) :: name
  check_nml_error = 0
end function check_nml_error

logical function file_exist(name)
  character(len=*), intent(in) :: name
  file_exist = .false.   ! no input files in the fixture sandbox -> namelist defaults
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
