!! Scaffold to fixture spectral_dynamics.F90's PRIVATE `compute_corrections`
!! routine (mass/energy/water conservation corrections). Same approach as
!! four_in_one_wrapper.F90: the subroutine body is extracted VERBATIM from the
!! pinned source (lines 1213-1302, byte-for-byte via sed) and dropped into this
!! thin module, which supplies the module-variable environment it references
!! (time-level state arrays psg/ug/vg/tg/ln_ps/ts, the correction flags and the
!! mean_*_previous references, spectral bounds, etc.). The global integrals are
!! the REAL routines: mass_weighted_global_integral (global_integral.F90) and
!! area_weighted_global_mean (transforms_grid_stub, which replicates
!! transforms.F90's formula verbatim). NO numerics are reimplemented here.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   sed -n '1213,1302p' $ISCA_SRC/src/atmos_spectral/model/spectral_dynamics.F90 \
!!       > compute_corrections_body.inc
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 mpp_domains_stub.F90 transforms_grid_stub.F90 tracer_type_stub.F90 \
!!     $ISCA_SRC/src/atmos_spectral/tools/gauss_and_legendre.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/press_and_geopot.F90 \
!!     $ISCA_SRC/src/atmos_spectral/model/global_integral.F90 \
!!     compute_corrections_wrapper.F90 jsca_dump.F90 \
!!     dump_compute_corrections_reference.F90 -o dump_compute_corrections_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_compute_corrections ./dump_compute_corrections_reference

module compute_corrections_mod
use          constants_mod, only: grav, cp_air
use                fms_mod, only: error_mesg, FATAL
use          transforms_mod, only: area_weighted_global_mean
use     global_integral_mod, only: mass_weighted_global_integral
use        tracer_type_mod, only: tracer_type
implicit none
public

integer :: ms, me, ns, ne, is, ie, js, je, num_levels, num_tracers, nhum
integer :: previous, current, future
logical :: do_mass_correction, do_energy_correction, do_water_correction, dry_model
real    :: mean_surf_press_previous, mean_energy_previous, mean_water_previous
real    :: water_correction_limit

real,    allocatable, dimension(:,:,:)     :: psg
real,    allocatable, dimension(:,:,:,:)   :: ug, vg, tg
complex, allocatable, dimension(:,:,:)     :: ln_ps
complex, allocatable, dimension(:,:,:,:)   :: ts
real,    allocatable, dimension(:,:,:,:,:) :: grid_tracers
complex, allocatable, dimension(:,:,:,:,:) :: spec_tracers

contains

include "compute_corrections_body.inc"

end module compute_corrections_mod
