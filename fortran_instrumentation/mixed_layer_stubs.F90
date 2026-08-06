!! Infrastructure stubs for compiling mixed_layer.F90 (+ vert_diff.F90) standalone.
!!
!! mixed_layer pulls in the whole model plumbing -- transforms, the spectral
!! dynamics (surface geopotential), mpp domains, the q-flux and SST interpolators,
!! diagnostics -- but for the Frierson slab-ocean step (land_option='none',
!! do_qflux=.false., prescribe_initial_dist=.true., no restart files) none of
!! that touches the numerics: the surface energy balance is a closed arithmetic
!! form of the fluxes and the vert_diff surface coupling. These stubs supply the
!! symbols so the real module compiles and runs unmodified. No numerics.
!!
!! (fms_mod / constants_mod come from fms_stubs.F90; mpp_mod from mpp_mod_stub.F90.)

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
  query_method = .false.
end function query_method
end module tracer_manager_mod

!-------------------------------------------------------------------------------
module time_manager_mod
implicit none
private
public :: time_type
type time_type
  integer :: seconds = 0, days = 0
end type time_type
end module time_manager_mod

!-------------------------------------------------------------------------------
module diag_manager_mod
use time_manager_mod, only: time_type
implicit none
private
public :: register_diag_field, register_static_field, send_data
interface send_data
  module procedure send_2d
end interface
contains
integer function register_diag_field(module_name, field_name, axes, init_time, &
                                     long_name, units, missing_value)
  character(len=*), intent(in) :: module_name, field_name
  integer, intent(in) :: axes(:)
  type(time_type), intent(in) :: init_time
  character(len=*), intent(in), optional :: long_name, units
  real, intent(in), optional :: missing_value
  register_diag_field = -1
end function register_diag_field
integer function register_static_field(module_name, field_name, axes, &
                                        long_name, units, missing_value)
  character(len=*), intent(in) :: module_name, field_name
  integer, intent(in) :: axes(:)
  character(len=*), intent(in), optional :: long_name, units
  real, intent(in), optional :: missing_value
  register_static_field = -1
end function register_static_field
logical function send_2d(id, field, time, is_in, js_in)
  integer, intent(in) :: id
  real, intent(in) :: field(:,:)
  type(time_type), intent(in), optional :: time
  integer, intent(in), optional :: is_in, js_in
  send_2d = .true.
end function send_2d
end module diag_manager_mod

!-------------------------------------------------------------------------------
module transforms_mod
implicit none
private
public :: get_deg_lat, get_deg_lon, grid_domain
integer :: grid_domain = 0
contains
subroutine get_deg_lat(deg_lat)
  real, intent(out) :: deg_lat(:)
  integer :: j, n
  n = size(deg_lat)
  ! evenly spaced latitudes -90..90 (only used to build rad_lat_2d; the Frierson
  ! step's numerics do not depend on latitude for a pure slab ocean)
  do j = 1, n
    deg_lat(j) = -90.0 + 180.0 * real(j-1) / real(max(n-1, 1))
  end do
end subroutine get_deg_lat
subroutine get_deg_lon(deg_lon)
  real, intent(out) :: deg_lon(:)
  integer :: i, n
  n = size(deg_lon)
  do i = 1, n
    deg_lon(i) = 360.0 * real(i-1) / real(n)
  end do
end subroutine get_deg_lon
end module transforms_mod

!-------------------------------------------------------------------------------
module spectral_dynamics_mod
implicit none
private
public :: get_surf_geopotential
contains
subroutine get_surf_geopotential(zsurf)
  real, intent(out) :: zsurf(:,:)
  zsurf = 0.0    ! link-only (land_option='zsurf' path, not exercised)
end subroutine get_surf_geopotential
end module spectral_dynamics_mod

!-------------------------------------------------------------------------------
module mpp_domains_mod
implicit none
private
public :: mpp_get_global_domain
contains
subroutine mpp_get_global_domain(domain, xsize, ysize)
  integer, intent(in) :: domain
  integer, intent(out), optional :: xsize, ysize
  if (present(xsize)) xsize = 0
  if (present(ysize)) ysize = 0
end subroutine mpp_get_global_domain
end module mpp_domains_mod

!-------------------------------------------------------------------------------
module interpolator_mod
use time_manager_mod, only: time_type
implicit none
private
public :: interpolate_type, interpolator_init, interpolator, CONSTANT
integer, parameter :: CONSTANT = 1
type interpolate_type
  integer :: dummy = 0
end type interpolate_type
interface interpolator
  module procedure interpolator_2d
end interface
contains
subroutine interpolator_init(clim_type, file_name, lonb, latb, data_out_of_bounds)
  type(interpolate_type), intent(inout) :: clim_type
  character(len=*), intent(in) :: file_name
  real, intent(in), dimension(:,:) :: lonb, latb
  integer, intent(in), dimension(:) :: data_out_of_bounds
end subroutine interpolator_init
subroutine interpolator_2d(clim_type, Time, interp_data, field_name)
  type(interpolate_type), intent(inout) :: clim_type
  type(time_type), intent(in) :: Time
  real, intent(out), dimension(:,:) :: interp_data
  character(len=*), intent(in) :: field_name
  interp_data = 0.0
end subroutine interpolator_2d
end module interpolator_mod

!-------------------------------------------------------------------------------
module qflux_mod
implicit none
private
public :: qflux_init, qflux, warmpool
contains
subroutine qflux_init()
end subroutine qflux_init
subroutine qflux(lat, flux)
  real, intent(in) :: lat(:)
  real, intent(inout) :: flux(:,:)
end subroutine qflux
subroutine warmpool(lon, lat, flux)
  real, intent(in) :: lon(:), lat(:)
  real, intent(inout) :: flux(:,:)
end subroutine warmpool
end module qflux_mod
