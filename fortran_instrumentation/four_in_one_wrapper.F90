!! Scaffold to fixture spectral_dynamics.F90's PRIVATE `four_in_one` routine.
!!
!! four_in_one is a private grid-space kernel of the (enormous, un-stubbable)
!! spectral_dynamics_mod. To exercise it unmodified, its subroutine body is
!! extracted VERBATIM from the pinned source (lines 1038-1112, byte-for-byte via
!! sed in the build recipe) and dropped into this thin module, which supplies the
!! module-variable environment it references (is/ie/js/je, num_levels,
!! vert_difference_option, dpk/dbk/bk) plus rdgas/cp_air from constants. NO
!! numerics are reimplemented here — only the variable declarations the routine
!! would otherwise inherit from spectral_dynamics_mod.
!!
!! Build (from repo root; ISCA_SRC = pinned Isca checkout):
!!   cd fortran_instrumentation
!!   sed -n '1038,1112p' $ISCA_SRC/src/atmos_spectral/model/spectral_dynamics.F90 \
!!       > four_in_one_body.inc
!!   gfortran -O2 -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none \
!!     fms_stubs.F90 four_in_one_wrapper.F90 jsca_dump.F90 \
!!     dump_four_in_one_reference.F90 -o dump_four_in_one_reference
!! Run:
!!   JSCA_DUMP_DIR=../tests/fixtures/raw_four_in_one ./dump_four_in_one_reference

module four_in_one_mod
use constants_mod, only: rdgas, cp_air
implicit none
public

integer :: is, ie, js, je, num_levels
character(len=64) :: vert_difference_option
real, allocatable, dimension(:) :: bk, dpk, dbk

contains

include "four_in_one_body.inc"

end module four_in_one_mod
