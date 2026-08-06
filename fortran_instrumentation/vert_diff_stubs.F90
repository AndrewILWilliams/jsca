!! field_manager / tracer_manager stubs for compiling vert_diff.F90 standalone.
!!
!! vert_diff uses these only for bookkeeping: how many prognostic tracers there
!! are and which index is specific humidity. The Frierson moist model carries a
!! single prognostic tracer, sphum. These stubs advertise exactly that
!! (num_prog=1, sphum=index 1, no land/ice exchange, default diffusion), so
!! vert_diff_init sets sphum's do_vert_diff=.false. and routes humidity through
!! the T/q path -- matching the real run. No numerics.

module field_manager_mod
implicit none
private
public :: MODEL_ATMOS, MODEL_LAND, MODEL_ICE
integer, parameter :: MODEL_ATMOS = 1, MODEL_LAND = 2, MODEL_ICE = 3
end module field_manager_mod

!-------------------------------------------------------------------------------
module tracer_manager_mod
use field_manager_mod, only: MODEL_ATMOS
implicit none
private
public :: query_method, get_number_tracers, get_tracer_index, &
          get_tracer_names, NO_TRACER
integer, parameter :: NO_TRACER = -99

contains

subroutine get_number_tracers(model, num_tracers, num_prog, num_diag, num_family)
  integer, intent(in) :: model
  integer, intent(out), optional :: num_tracers, num_prog, num_diag, num_family
  ! the Frierson moist model: a single prognostic tracer (sphum)
  if (present(num_tracers)) num_tracers = 1
  if (present(num_prog))    num_prog = 1
  if (present(num_diag))    num_diag = 0
  if (present(num_family))  num_family = 0
end subroutine get_number_tracers

integer function get_tracer_index(model, name, indices)
  integer, intent(in) :: model
  character(len=*), intent(in) :: name
  integer, intent(in), optional :: indices(:)
  if (trim(name) == 'sphum' .and. model == MODEL_ATMOS) then
    get_tracer_index = 1
  else
    get_tracer_index = NO_TRACER
  end if
end function get_tracer_index

subroutine get_tracer_names(model, n, name, longname, units, err_msg)
  integer, intent(in) :: model, n
  character(len=*), intent(out) :: name
  character(len=*), intent(out), optional :: longname, units, err_msg
  name = 'sphum'
  if (present(longname)) longname = 'specific humidity'
  if (present(units))    units = 'kg/kg'
  if (present(err_msg))  err_msg = ''
end subroutine get_tracer_names

logical function query_method(method, model, n, name, control)
  character(len=*), intent(in) :: method
  integer, intent(in) :: model, n
  character(len=*), intent(out) :: name
  character(len=*), intent(out), optional :: control
  name = ''
  if (present(control)) control = ''
  query_method = .false.       ! -> default diffusion for every tracer
end function query_method
end module tracer_manager_mod
