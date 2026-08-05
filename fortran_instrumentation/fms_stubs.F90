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

end module fms_mod
