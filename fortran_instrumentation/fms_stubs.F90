!! Minimal stubs for fms_mod / constants_mod so numerical kernels from the
!! Isca tree (e.g. gauss_and_legendre.F90) compile standalone, without FMS.
!! Only logging/bookkeeping is stubbed — no numerics live here.

module constants_mod
implicit none
real, public, parameter :: pi = 3.14159265358979323846
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

subroutine error_mesg(routine, message, level)
  character(len=*), intent(in) :: routine, message
  integer, intent(in) :: level
  write(*,*) 'ERROR (', trim(routine), '): ', trim(message)
  if (level == FATAL) stop 1
end subroutine error_mesg

subroutine write_version_number(version, tagname)
  character(len=*), intent(in) :: version, tagname
end subroutine write_version_number

end module fms_mod
