"""Physical constants, ported verbatim from Isca's ``src/shared/constants/constants.F90``.

Provenance
----------
Every value below is copied from the Fortran source in the pinned Isca snapshot
(``Iscamaster 2.zip``). Isca compiles with ``-fdefault-real-8`` (see
``mkmf.template.ubuntu_conda``), so all Fortran ``real`` parameters are 64-bit;
Python floats match exactly.

Notes
-----
* Isca makes several "constants" *runtime-configurable* via ``constants_nml``
  (``RADIUS``, ``OMEGA``, ``GRAV``, ``RDGAS``, ``KAPPA``, ``PSTD`` … are
  ``real, public ::`` variables, not ``parameter``). In jsca those live in the
  model config; the values here are Isca's defaults and the Earth references.
* Do not "correct" values that look off (e.g. ``RADIUS = 6376 km``, not 6371;
  ``GRAV = 9.80``, not 9.81). Reproducing Isca means reproducing its constants.
"""

# --- Earth reference values (constants.F90 lines ~83-86, 240-256) -----------
EARTH_GRAV = 9.80  # m/s^2  (sic — Isca's value)
EARTH_RDGAS = 287.04  # J/kg/K
EARTH_KAPPA = 2.0 / 7.0
EARTH_CP_AIR = EARTH_RDGAS / EARTH_KAPPA  # = 1004.64
EARTH_OMEGA = 7.2921150e-5  # s^-1
EARTH_RADIUS = 6376.0e3  # m  (sic — Isca's default, namelist-overridable)
PSTD_MKS_EARTH = 101325.0  # Pa

# Defaults for the mutable planetary constants (namelist-overridable in Isca).
RADIUS = EARTH_RADIUS
OMEGA = EARTH_OMEGA
GRAV = EARTH_GRAV
RDGAS = EARTH_RDGAS
KAPPA = EARTH_KAPPA
CP_AIR = EARTH_CP_AIR
PSTD = 1.013250e6  # dynes/cm^2
PSTD_MKS = 101325.0  # Pa

# --- Water / moist thermodynamics -------------------------------------------
RVGAS = 461.50  # J/kg/K
CP_VAPOR = 4.0 * RVGAS
DENS_H2O = 1000.0  # kg/m^3
HLV = 2.500e6  # J/kg latent heat of evaporation
HLF = 3.34e5  # J/kg latent heat of fusion
HLS = HLV + HLF  # J/kg latent heat of sublimation
TFREEZE = 273.16  # K
DEF_ES0 = 1.0

# --- Ocean (slab / mixed layer) ----------------------------------------------
CP_OCEAN = 3989.24495292815  # J/kg/K
RHO0 = 1.035e3  # kg/m^3
RHO0R = 1.0 / RHO0
RHO_CP = RHO0 * CP_OCEAN

# --- Radiation / molecular weights -------------------------------------------
WTMAIR = 2.896440e01
WTMH2O = WTMAIR * (EARTH_RDGAS / RVGAS)
WTMOZONE = 47.99820
WTMC = 12.00000
WTMCO2 = 44.00995
WTMO2 = 31.9988
WTMCFC11 = 137.3681
WTMCFC12 = 120.9135
DIFFAC = 1.660000e00
AVOGNO = 6.023000e23
RADCON = ((1.0e02 * EARTH_GRAV) / (1.0e04 * EARTH_CP_AIR)) * 8.640000e04
RADCON_MKS = (EARTH_GRAV / EARTH_CP_AIR) * 8.640000e04
O2MIXRAT = 2.0953e-01
RHOAIR = 1.292269  # kg/m^3
ALOGMIN = -50.0
GAS_CONSTANT = 8.314  # J/mol/K (sic — Isca's value, see source comment)

# --- Misc ---------------------------------------------------------------------
SECONDS_PER_DAY = 8.640000e04
SECONDS_PER_HOUR = 3600.0
SECONDS_PER_MINUTE = 60.0
KELVIN = 273.15
STEFAN = 5.6734e-8  # W/m^2/K^4 (sic — Isca's value)
VONKARM = 0.40
PI = 3.14159265358979323846
RAD_TO_DEG = 180.0 / PI
DEG_TO_RAD = PI / 180.0
