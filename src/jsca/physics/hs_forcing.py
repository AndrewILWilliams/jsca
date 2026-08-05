"""Held & Suarez (1994) idealized forcing.

Faithful port of the numerical core of Isca's
``src/atmos_param/hs_forcing/hs_forcing.F90`` — the Newtonian relaxation of
temperature toward a radiative-equilibrium profile (``newtonian_damping``, F90
L508) and the Rayleigh damping of the near-surface winds (``rayleigh_damping``,
F90 L615), assembled with the frictional-heating energy term (F90 L197-199) as
the public ``hs_forcing`` does (F90 L148). This is the physics forcing of the
Held–Suarez benchmark that couples to the spectral dynamical core.

Scope: the default Isca configuration — ``equilibrium_t_option = 'Held_Suarez'``,
``do_conserve_energy = .true.``, ``no_forcing = .false.``, no
``relax_to_specified_wind``, no local heating, no tracers. The other options
(``from_file`` / ``exoplanet`` / ``top_down`` equilibrium temperature, specified
wind relaxation, local heating, the optional tracer source/sink) pull in Isca's
interpolator / astronomy / diagnostics / tracer-manager infrastructure and are
not ported here; they are documented as follow-ups.

Layout: column physics, **level axis last**. ``t``, ``u``, ``v``, ``p_full`` are
``(..., K)``; ``p_half`` is ``(..., K+1)``; ``lat`` is ``(...)`` (broadcast over
levels). Leading axes are batched. Surface pressure is the bottom half-level,
``ps = p_half[..., -1]`` (F90 L190).

Coefficients (F90 L391-404): ``ka``/``ks``/``kf`` are damping *times* in days
when negative (the Isca default ``-40``/``-4``/``-1``), converted to rates
``tka = -1/(86400*ka)`` etc. Note the literal ``86400`` (not
``SECONDS_PER_DAY``) — reproduced.

Fixtures: ``tests/fixtures/hs_forcing_reference.npz`` from
``fortran_instrumentation/dump_hs_forcing_reference.F90``, which compiles the
real ``hs_forcing.F90`` unmodified against no-op stubs for its FMS
infrastructure. The fixture validates the composite ``(udt, vdt, tdt)`` returned
by the public ``hs_forcing`` (``teq`` is a Fortran-internal diagnostic, implied
by ``tdt``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jsca import constants

Array = jnp.ndarray


@dataclass(frozen=True)
class HsForcingParams:
    """Static Held–Suarez configuration (hashable; pass as a static jit arg).

    Scalars mirror the ``hs_forcing_nml`` defaults (F90 L77-80). ``tka``/``tks``/
    ``vkf`` are the converted damping rates (1/s).
    """

    t_zero: float = 315.0
    t_strat: float = 200.0
    delh: float = 60.0
    delv: float = 10.0
    eps: float = 0.0
    sigma_b: float = 0.7
    p00: float = 1.0e5
    tka: float = 1.0 / (86400.0 * 40.0)  # ka = -40 days
    tks: float = 1.0 / (86400.0 * 4.0)  # ks = -4 days
    vkf: float = 1.0 / (86400.0 * 1.0)  # kf = -1 day
    do_conserve_energy: bool = True
    kappa: float = constants.KAPPA
    cp_air: float = constants.CP_AIR


def hs_forcing_init(
    ka: float = -40.0,
    ks: float = -4.0,
    kf: float = -1.0,
    **kwargs: float,
) -> HsForcingParams:
    """Build :class:`HsForcingParams`, converting ``ka``/``ks``/``kf`` (F90 L391-404).

    A negative value is a damping *time in days* (converted to a 1/s rate); a
    positive value is already a rate. Remaining scalars may be overridden via
    ``kwargs`` (e.g. ``t_zero=315.0``).
    """
    tka = -1.0 / (86400.0 * ka) if ka < 0.0 else ka
    tks = -1.0 / (86400.0 * ks) if ks < 0.0 else ks
    vkf = -1.0 / (86400.0 * kf) if kf < 0.0 else kf
    return HsForcingParams(tka=tka, tks=tks, vkf=vkf, **kwargs)


def newtonian_damping(
    p: HsForcingParams, lat: Array, ps: Array, p_full: Array, t: Array
) -> tuple[Array, Array]:
    """Thermal relaxation toward radiative equilibrium — port of ``newtonian_damping``
    (Held_Suarez branch, F90 L508-611). Returns ``(tdt, teq)``, both ``(..., K)``.
    """
    sin_lat = jnp.sin(lat)
    cos_lat_2 = 1.0 - sin_lat * sin_lat
    cos_lat_4 = cos_lat_2 * cos_lat_2
    t_star = p.t_zero - p.delh * sin_lat * sin_lat - p.eps * sin_lat  # (...)
    tstr = p.t_strat - p.eps * sin_lat  # (...)
    tcoeff = (p.tks - p.tka) / (1.0 - p.sigma_b)

    p_norm = p_full / p.p00  # (..., K)
    the = t_star[..., None] - p.delv * cos_lat_2[..., None] * jnp.log(p_norm)
    teq = the * p_norm**p.kappa
    teq = jnp.maximum(teq, tstr[..., None])  # F90 L570

    sigma = p_full / ps[..., None]
    tfactr = tcoeff * (sigma - p.sigma_b)
    in_pbl = (sigma <= 1.0) & (sigma > p.sigma_b)
    tdamp = jnp.where(in_pbl, p.tka + cos_lat_4[..., None] * tfactr, p.tka)  # F90 L591-596

    tdt = -tdamp * (t - teq)  # F90 L601
    return tdt, teq


def rayleigh_damping(
    p: HsForcingParams, ps: Array, p_full: Array, u: Array, v: Array
) -> tuple[Array, Array]:
    """Rayleigh drag of the near-surface winds — port of ``rayleigh_damping``
    (F90 L615-679). Returns ``(udt, vdt)``, both ``(..., K)``.
    """
    vcoeff = -p.vkf / (1.0 - p.sigma_b)
    sigma = p_full / ps[..., None]
    vfactr = vcoeff * (sigma - p.sigma_b)
    in_pbl = (sigma <= 1.0) & (sigma > p.sigma_b)  # F90 L660
    udt = jnp.where(in_pbl, vfactr * u, 0.0)
    vdt = jnp.where(in_pbl, vfactr * v, 0.0)
    return udt, vdt


def hs_forcing(
    p: HsForcingParams,
    lat: Array,
    p_half: Array,
    p_full: Array,
    u: Array,
    v: Array,
    t: Array,
    um: Array,
    vm: Array,
    dt: float,
) -> tuple[Array, Array, Array, Array]:
    """Composite Held–Suarez forcing tendencies — port of the ``hs_forcing`` driver
    (default path, F90 L148-268). Returns ``(udt, vdt, tdt, teq)``.

    ``udt``/``vdt`` are the Rayleigh drag; ``tdt`` is the Newtonian heating plus
    the Rayleigh frictional-heating term (``do_conserve_energy``); ``teq`` is the
    equilibrium temperature (diagnostic). Fields are ``(..., K)``; ``p_half`` is
    ``(..., K+1)``; ``lat`` is ``(...)``. ``um``/``vm`` are the previous-step
    winds used by the energy-conserving heating (F90 L197).
    """
    ps = p_half[..., -1]  # surface pressure (F90 L190)
    utnd, vtnd = rayleigh_damping(p, ps, p_full, u, v)

    ttnd, teq = newtonian_damping(p, lat, ps, p_full, t)
    if p.do_conserve_energy:
        # frictional heating from the Rayleigh drag (F90 L197)
        ttnd_fric = -((um + 0.5 * utnd * dt) * utnd + (vm + 0.5 * vtnd * dt) * vtnd) / p.cp_air
        tdt = ttnd + ttnd_fric
    else:
        tdt = ttnd
    return utnd, vtnd, tdt, teq
