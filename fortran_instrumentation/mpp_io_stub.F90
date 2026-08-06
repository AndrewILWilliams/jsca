!! Minimal mpp_io_mod stub. sat_vapor_pres.F90 uses only mpp_close (on the
!! non-INTERNAL_FILE_NML namelist path, which the fixture never takes). Supplies
!! the symbol so the `use` resolves. No I/O, no numerics.

module mpp_io_mod
implicit none
private
public :: mpp_close

contains

subroutine mpp_close(unit)
  integer, intent(in) :: unit
end subroutine mpp_close

end module mpp_io_mod
