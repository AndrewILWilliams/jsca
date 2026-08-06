"""Top-of-model Rayleigh sponge — Isca's ``damping_driver.F90``.

Faithful port of the only ``damping_driver`` path the Frierson aquaplanet
activates: the **Rayleigh sponge** (``damping_driver_nml: do_rayleigh=.true.,
trayfric=-0.25, sponge_pbottom=5000., do_conserve_energy=.true.``). The three
gravity-wave-drag paths (``do_mg_drag``/``do_cg_drag``/``do_topo_drag``) and the
constant-drag path are all off for Frierson and are **not** ported here; the
docstring flags that scope.

**What the sponge does (F90 ``rayleigh``, L594-637).** Near the model top the
spectral core has no physical dissipation, so upward-propagating waves reflect off
the lid and contaminate the circulation. The sponge relaxes the horizontal winds
toward zero in the topmost layers, with a rate that grows quadratically as the
pressure drops below ``sponge_pbottom``::

    fact = rfactr · (sponge_pbottom − p_full)² / sponge_pbottom²      (F90 L610)
    du/dt = −u · fact ;  dv/dt = −v · fact                            (F90 L611-612)

applied only where ``p_full < sponge_pbottom`` **and** only over the top
``nlev_rayfric`` levels (F90 L607-614). ``rfactr`` is the sponge rate at the lid:
``1/trayfric`` if ``trayfric`` is a time in seconds, else ``(1/|trayfric|)/86400``
when ``trayfric`` is given as a (negative) time in days (F90 L415-419, with the
module parameter ``daypsec = 1/86400``).

**Frictional heating (F90 L624-628).** With ``do_conserve_energy=.true.`` the
kinetic energy the drag removes reappears as a heating, using the *time-centred*
wind ``(u + ½·dt·du/dt)`` so the discrete energy budget closes::

    dT/dt = −[ (u + ½·dt·du/dt)·du/dt + (v + ½·dt·dv/dt)·dv/dt ] / cp_air

Since ``du/dt = dv/dt = 0`` outside the sponge, ``dT/dt`` is automatically zero
there, so the whole-array form reproduces the Fortran's ``k = 1 … nlev_rayfric``
loop exactly.

**``nlev_rayfric`` (F90 L413-414).** The sponge depth is set once at init from the
reference full-level pressures ``pref``: ``nlev_rayfric`` is the level whose
``pref`` is closest to ``2·sponge_pbottom`` (Fortran ``minloc`` is 1-based, so it
is a *count* of top levels). :func:`damping_driver_init` computes it.

Layout: column fields ``(..., K)`` with ``k = 0`` the model top … ``K−1`` the
surface (Isca's ``k = 1`` top). Pure arithmetic — no documented deviation.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from jsca import constants

# damping_driver.F90 module parameter ``daypsec = 1./86400`` (L99) — a literal in
# the Fortran, not constants_mod's seconds_per_day; kept literal to match.
_DAYPSEC = 1.0 / 86400.0


@dataclass(frozen=True)
class DampingDriverParams:
    """Static ``damping_driver_nml`` sponge configuration (hashable; static jit arg).

    ``nlev_rayfric`` and ``rfactr`` are the init-time derived quantities (see
    :func:`damping_driver_init`); the rest mirror the namelist.
    """

    nlev_rayfric: int            # number of top levels the sponge acts on (F90 count)
    rfactr: float                # sponge rate at the lid (1/s)
    sponge_pbottom: float = 5000.0   # sponge base pressure (Pa)
    do_conserve_energy: bool = True


def damping_driver_init(pref, trayfric: float = -0.25,
                        sponge_pbottom: float = 5000.0,
                        do_conserve_energy: bool = True) -> DampingDriverParams:
    """Build sponge params from the reference pressure profile (F90 L411-420).

    ``pref`` is the 1-D reference full-level pressure profile (top→surface, Pa;
    Isca passes ``pref(1:num_levels)`` from ``pressure_variables`` at
    ``PSTD_MKS``). ``nlev_rayfric`` is the 1-based index of the level whose
    ``pref`` is closest to ``2·sponge_pbottom`` — i.e. the number of top levels
    the sponge spans. ``rfactr`` converts ``trayfric`` (seconds if positive, days
    if negative) to the lid rate.
    """
    import numpy as np

    pref = np.asarray(pref, dtype=float)
    # Fortran: raylev = minloc(abs(pref - 2*sponge_pbottom)); nlev_rayfric = raylev(1).
    # minloc is 1-based, so add 1 to the 0-based argmin to recover the level count.
    nlev_rayfric = int(np.argmin(np.abs(pref - 2.0 * sponge_pbottom))) + 1
    if trayfric > 0.0:
        rfactr = 1.0 / trayfric
    else:
        rfactr = (1.0 / abs(trayfric)) * _DAYPSEC
    return DampingDriverParams(nlev_rayfric=nlev_rayfric, rfactr=rfactr,
                               sponge_pbottom=sponge_pbottom,
                               do_conserve_energy=do_conserve_energy)


def rayleigh_sponge(params: DampingDriverParams, dt, p_full, u, v):
    """One Rayleigh-sponge step — port of ``rayleigh`` (F90 L594-637).

    Args (``(..., K)`` columns, ``k = 0`` top): the full-level pressure and the
    horizontal winds. ``dt`` is the (leapfrog) timestep used by the energy-
    conserving heating.

    Returns ``(udt, vdt, tdt)`` — the wind tendencies (relaxation toward zero) and
    the frictional heating (zero unless ``do_conserve_energy``). These *accumulate*
    into the physics tendencies in the driver (F90 L169-171).
    """
    k_idx = jnp.arange(u.shape[-1])
    # sponge acts where p_full < sponge_pbottom AND within the top nlev_rayfric levels
    in_sponge = (p_full < params.sponge_pbottom) & (k_idx < params.nlev_rayfric)
    fact = params.rfactr * (params.sponge_pbottom - p_full) ** 2 / params.sponge_pbottom ** 2
    fact = jnp.where(in_sponge, fact, 0.0)

    udt = -u * fact
    vdt = -v * fact

    if params.do_conserve_energy:
        # time-centred KE dissipation -> heating; zero where udt=vdt=0, so this
        # whole-array form matches the Fortran k=1..nlev_rayfric loop exactly.
        tdt = -((u + 0.5 * dt * udt) * udt + (v + 0.5 * dt * vdt) * vdt) / constants.CP_AIR
    else:
        tdt = jnp.zeros_like(u)
    return udt, vdt, tdt
