!! No-op infrastructure stubs so hs_forcing.F90 compiles/links standalone for the
!! Held-Suarez fixture driver. hs_forcing pulls in a lot of FMS machinery
!! (time/diag/field/tracer managers, interpolator, astronomy, domains) but the
!! default Held_Suarez path exercises none of its numerics — only the two core
!! routines (newtonian_damping, rayleigh_damping) and the frictional-heating
!! term run. Everything here is link-only or a trivial no-op; NO numerics.
!!
!! Design for the default path: file_exist -> .false. (namelist defaults used),
!! register_diag_field -> -1 (every diagnostic send is guarded off),
!! get_number_tracers -> 0 (the tracer loop is zero-iteration), get_grid_domain
!! -> full serial grid. grid_domain is a plain integer here (only set_domain uses
!! it, and both agree on the type).

module time_manager_mod
implicit none
private
public :: time_type, get_time
type time_type
  integer :: seconds = 0, days = 0
end type time_type
contains
subroutine get_time(t, seconds, days)
  type(time_type), intent(in) :: t
  integer, intent(out) :: seconds, days
  seconds = t%seconds
  days = t%days
end subroutine get_time
end module time_manager_mod

!-------------------------------------------------------------------------------

module diag_manager_mod
use time_manager_mod, only: time_type
implicit none
private
public :: register_diag_field, send_data

interface send_data
  module procedure send_data_2d, send_data_3d
end interface

contains

integer function register_diag_field(module_name, field_name, axes, init_time, &
                                     long_name, units, missing_value, range)
  character(len=*), intent(in) :: module_name, field_name
  integer, intent(in) :: axes(:)
  type(time_type), intent(in), optional :: init_time
  character(len=*), intent(in), optional :: long_name, units
  real, intent(in), optional :: missing_value
  real, intent(in), optional :: range(:)
  register_diag_field = -1   ! disabled -> all send_data calls are guarded off
end function register_diag_field

logical function send_data_2d(id, field, time)
  integer, intent(in) :: id
  real, intent(in) :: field(:,:)
  type(time_type), intent(in) :: time
  send_data_2d = .true.
end function send_data_2d

logical function send_data_3d(id, field, time)
  integer, intent(in) :: id
  real, intent(in) :: field(:,:,:)
  type(time_type), intent(in) :: time
  send_data_3d = .true.
end function send_data_3d

end module diag_manager_mod

!-------------------------------------------------------------------------------

module field_manager_mod
implicit none
private
public :: MODEL_ATMOS, parse
integer, parameter :: MODEL_ATMOS = 1
contains
integer function parse(text, name, value)
  character(len=*), intent(in) :: text, name
  real, intent(out) :: value
  value = 0.0
  parse = 0
end function parse
end module field_manager_mod

!-------------------------------------------------------------------------------

module tracer_manager_mod
implicit none
private
public :: query_method, get_number_tracers
contains
subroutine get_number_tracers(model, num_tracers, num_prog, num_diag, num_family)
  integer, intent(in) :: model
  integer, intent(out), optional :: num_tracers, num_prog, num_diag, num_family
  if (present(num_tracers)) num_tracers = 0
  if (present(num_prog))    num_prog = 0
  if (present(num_diag))    num_diag = 0
  if (present(num_family))  num_family = 0
end subroutine get_number_tracers

logical function query_method(method, model, n, name, control)
  character(len=*), intent(in) :: method
  integer, intent(in) :: model, n
  character(len=*), intent(out) :: name
  character(len=*), intent(out), optional :: control
  name = ''
  if (present(control)) control = ''
  query_method = .false.
end function query_method
end module tracer_manager_mod

!-------------------------------------------------------------------------------

module interpolator_mod
implicit none
private
public :: interpolate_type, interpolator_init, interpolator, interpolator_end
public :: CONSTANT, INTERP_WEIGHTED_P
integer, parameter :: CONSTANT = 1, INTERP_WEIGHTED_P = 2
type interpolate_type
  integer :: dummy = 0
end type interpolate_type
contains
subroutine interpolator_init(clim_type, file_name, lonb, latb, data_out_of_bounds, vert_interp)
  type(interpolate_type), intent(inout) :: clim_type
  character(len=*), intent(in) :: file_name
  real, intent(in) :: lonb(:,:), latb(:,:)
  integer, intent(in), optional :: data_out_of_bounds(:)
  integer, intent(in), optional :: vert_interp(:)
end subroutine interpolator_init

subroutine interpolator(clim_type, p_half, interp_data, field_name)
  type(interpolate_type), intent(inout) :: clim_type
  real, intent(in) :: p_half(:,:,:)
  real, intent(out) :: interp_data(:,:,:)
  character(len=*), intent(in) :: field_name
  interp_data = 0.0
end subroutine interpolator

subroutine interpolator_end(clim_type)
  type(interpolate_type), intent(inout) :: clim_type
end subroutine interpolator_end
end module interpolator_mod

!-------------------------------------------------------------------------------

module astronomy_mod
use time_manager_mod, only: time_type
implicit none
private
public :: diurnal_exoplanet, astronomy_init, obliq, ecc
real, save :: obliq = 23.439   ! link-only (top_down/exoplanet paths, not run)
real, save :: ecc   = 0.0167
contains
subroutine astronomy_init()
end subroutine astronomy_init

subroutine diurnal_exoplanet(lat, lon, Time, coszen, fracday, rrsun)
  real, intent(in) :: lat(:,:), lon(:,:)
  type(time_type), intent(in) :: Time
  real, intent(out) :: coszen(:,:), fracday(:,:)
  real, intent(out) :: rrsun
  coszen = 0.0
  fracday = 0.0
  rrsun = 1.0
end subroutine diurnal_exoplanet
end module astronomy_mod

!-------------------------------------------------------------------------------

! transforms_mod: hs_forcing pulls only grid_domain + get_grid_domain (serial
! full grid). grid_domain is a bare integer (set_domain agrees on the type).
module transforms_mod
implicit none
private
public :: grid_domain, get_grid_domain
integer, save :: grid_domain = 0
integer, save :: g_nlon = 0, g_nlat = 0
public :: transforms_grid_init
contains
subroutine transforms_grid_init(nlon, nlat)
  integer, intent(in) :: nlon, nlat
  g_nlon = nlon
  g_nlat = nlat
end subroutine transforms_grid_init

subroutine get_grid_domain(is, ie, js, je)
  integer, intent(out) :: is, ie, js, je
  is = 1
  ie = g_nlon
  js = 1
  je = g_nlat
end subroutine get_grid_domain
end module transforms_mod
