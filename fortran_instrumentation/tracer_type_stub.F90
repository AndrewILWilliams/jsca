!! Minimal tracer_type_mod stub: compute_corrections only reads
!! %numerical_representation, and only in the (dry-model, unused) water branch.
module tracer_type_mod
implicit none
public :: tracer_type
type tracer_type
  character(len=64) :: numerical_representation = 'grid'
end type tracer_type
end module tracer_type_mod
