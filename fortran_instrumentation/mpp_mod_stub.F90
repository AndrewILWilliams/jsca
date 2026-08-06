!! Minimal mpp_mod stub for vert_advection.F90 (serial, single-PE build).
!!
!! vert_advection uses mpp_sum/mpp_max/mpp_pe/mpp_sync only inside
!! vert_advection_end (CFL diagnostics), which the fixture driver never calls;
!! these supply the symbols so the `use` resolves. No numerics.
!!
!! (Named mpp_mod_stub to avoid clashing with mpp_domains_* stubs; do not compile
!! two modules of the same name in one build.)

module mpp_mod
implicit none
private
public :: mpp_sum, mpp_max, mpp_pe, mpp_sync
public :: input_nml_file

! FMS's internal-file namelist buffer. In the real FMS this holds the whole
! input.nml as an array of lines; here the fixture driver allocates and fills it
! before calling a module's *_init so that `read(input_nml_file, nml=...)` (under
! -DINTERNAL_FILE_NML) picks up the injected namelist groups. No numerics.
character(len=256), dimension(:), allocatable :: input_nml_file

interface mpp_sum
  module procedure mpp_sum_i, mpp_sum_r
end interface

interface mpp_max
  module procedure mpp_max_r
end interface

contains

integer function mpp_pe()
  mpp_pe = 0
end function mpp_pe

subroutine mpp_sync()
end subroutine mpp_sync

subroutine mpp_sum_i(x)
  integer, intent(inout) :: x
end subroutine mpp_sum_i

subroutine mpp_sum_r(x)
  real, intent(inout) :: x
end subroutine mpp_sum_r

subroutine mpp_max_r(x)
  real, intent(inout) :: x
end subroutine mpp_max_r

end module mpp_mod
