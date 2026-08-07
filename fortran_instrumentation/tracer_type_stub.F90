!! Minimal tracer_type_mod stub: compute_corrections reads
!! %numerical_representation (dry-model water branch); the update_tracers grid
!! branch reads %robert_coeff (the tracer's Robert/RAW filter coefficient).
module tracer_type_mod
implicit none
public :: tracer_type
type tracer_type
  character(len=64) :: numerical_representation = 'grid'
  real              :: robert_coeff = 0.0
end type tracer_type
end module tracer_type_mod
