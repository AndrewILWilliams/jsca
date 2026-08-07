!! Scaffold to fixture the GRID branch of spectral_dynamics.F90's private
!! `update_tracers` (the moist tracer time-step: horizontal + vertical advection
!! then the Robert/RAW leapfrog filter). Same verbatim-body approach as
!! compute_corrections_wrapper.F90: the grid-branch statements are extracted
!! BYTE-FOR-BYTE from the pinned source (lines 1224-1248 via sed) into
!! update_tracers_grid_body.inc and dropped into the thin subroutine below, which
!! supplies the module-variable environment they reference (the time-level state
!! ug/vg/grid_tracers, the step counters, raw_filter_coeff, the per-tracer vert
!! scheme, and the previous/current/future indices). The advection is the REAL
!! ported Fortran: a_grid_horiz_advection (fv_advection.F90) and vert_advection
!! (vert_advection.F90). NO numerics are reimplemented here.
!!
!! Only the grid branch is compiled (Frierson's sphum is a grid tracer), so none
!! of the spectral-branch machinery (transforms, compute_spectral_damping,
!! leapfrog on spectral coefficients) is needed and no stubs for it are required.
!!
!! Build (from this directory; ISCA_SRC = pinned Isca checkout):
!!   sed -n '1224,1248p' $ISCA_SRC/src/atmos_spectral/model/spectral_dynamics.F90 \
!!       > update_tracers_grid_body.inc
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 mpp_domains_fv_stub.F90 mpp_mod_stub.F90 tracer_type_stub.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/fv_advection.F90 \
!!     $ISCA_SRC/src/atmos_shared/vert_advection/vert_advection.F90 \
!!     update_tracers_wrapper.F90 jsca_dump.F90 \
!!     dump_update_tracers_reference.F90 -o dump_update_tracers_reference

module update_tracers_mod
use fv_advection_mod, only: a_grid_horiz_advection
use vert_advection_mod, only: vert_advection, ADVECTIVE_FORM
use     tracer_type_mod, only: tracer_type
implicit none
public

integer :: is, ie, js, je, num_levels, num_tracers
integer :: previous, current, future, step_number, num_steps
real    :: raw_filter_coeff
logical :: robert_complete_for_tracers
integer, allocatable, dimension(:)       :: tracer_vert_advect_scheme
real,    allocatable, dimension(:,:,:,:)   :: ug, vg
real,    allocatable, dimension(:,:,:,:,:) :: grid_tracers

contains

!! The grid branch of update_tracers, verbatim (fixed ntr = 1: one prognostic
!! tracer, as Frierson has). Args mirror the update_tracers signature subset the
!! grid branch touches; part_filt_tr_out is the RAW partially-filtered increment.
subroutine update_grid_tracer_ref(tracer_attributes, dt_tr, wg, p_half, delta_t, part_filt_tr_out)
  type(tracer_type), intent(inout), dimension(:)     :: tracer_attributes
  real,              intent(inout), dimension(:,:,:,:) :: dt_tr
  real,              intent(in),    dimension(:,:,:)   :: wg, p_half
  real,              intent(in)                        :: delta_t
  real,              intent(out),   dimension(is:ie, js:je, num_levels, num_tracers) :: part_filt_tr_out

  real, dimension(is:ie, js:je, num_levels) :: dp, dt_tmp, tr_future
  integer :: ntr

  ntr = 1
  include "update_tracers_grid_body.inc"
end subroutine update_grid_tracer_ref

end module update_tracers_mod
