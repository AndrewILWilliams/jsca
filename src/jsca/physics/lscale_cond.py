"""Large-scale (stratiform) condensation — Isca's ``do_simple`` path.

Faithful port of ``src/atmos_param/lscale_cond/lscale_cond.F90`` for the
Frierson moist aquaplanet, which runs ``lscale_cond_nml: do_simple=.true.,
do_evap=.true.`` (``exp/test_cases/frierson/frierson_test_case.py`` L117-120),
with the namelist default ``hc = 1.0``.

The scheme removes any supersaturation column by column. Where the air is
supersaturated it condenses just enough vapor to return to saturation, warming
the layer by the released latent heat; the condensate falls as rain, and
(``do_evap``) re-evaporates into sub-saturated layers on the way down.

**Algorithm (F90 L127-208), do_simple, no snow, no mask/conv** — the Frierson
driver calls ``lscale_cond`` without the optional ``mask``/``conv`` arguments
(``idealized_moist_phys.F90`` L981)::

    hlcp  = HLv / Cp_Air                                       (do_simple, L134)
    qsat, dqsat = compute_qs(tin, pfull, hc=hc)                (L145)
    do_adjust   = (qin - qsat)*qsat > 0                        (L155; supersat.)
    where do_adjust:                                           (L164-166)
        qdel = (qsat - qin)/(1 + hlcp*dqsat)      (< 0, drying)
        tdel = -hlcp*qdel                         (> 0, warming)
    elsewhere: qdel = tdel = 0
    pmass(k) = (phalf(k+1) - phalf(k))/Grav                    (L175)
    if do_evap: precip_evap(...)          re-evaporate falling rain (L216-255)
    precip = max(-sum_k pmass(k)*qdel(k), 0)                   (L190-194)
    rain   = precip                                           (do_simple, L198)

``tdel``/``qdel`` are **increments** over the physics step (not rates); the
driver divides by ``delta_t`` afterwards (``idealized_moist_phys.F90`` L987-988).
``rain`` is a column-integrated mass (kg/m^2).

**precip_evap (F90 L216-255), top → surface single pass** carrying the falling
rain mass ``exq`` (kg/m^2)::

    exq = 0
    for k = top .. surface:
        where qdel(k) < 0:  exq += -qdel(k)*pmass(k)      accumulate condensate
        where qdel(k) >= 0 and exq > 0:                  sub-saturated + rain aloft
            def = (qsat - qin)/(1 + hlcp*dqsat)           saturation deficit
            def = min(max(def, 0), exq/pmass(k))          limited by available rain
            qdel(k) += def ;  tdel(k) -= def*hlcp          moisten + cool
            exq     -= def*pmass(k)                        consume rain

Because ``qdel <= 0`` everywhere after the adjustment step, the two ``where``
branches are mutually exclusive per level (condensing vs. not), so the update is
unambiguous.

Layout: column physics, **level axis last**, k = 0 top … K-1 surface. ``tin``,
``qin``, ``pfull`` are ``(..., K)``; ``phalf`` is ``(..., K+1)``. Matches Isca's
half-level convention ``pmass(k) = (phalf(k+1) - phalf(k))/Grav`` directly.

Deviation: inherits only the documented ``sat_vapor_pres`` table-vs-closed-form
deviation via :func:`saturation_specific_humidity_and_deriv`; the condensation
arithmetic itself is exact.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from jsca import constants
from jsca.physics.sat_vapor_pres import saturation_specific_humidity_and_deriv

Array = jnp.ndarray


def _precip_evap(pmass, tin, qin, qsat, dqsat, hlcp, tdel, qdel):
    """Re-evaporate falling rain into sub-saturated layers (F90 ``precip_evap``).

    Single top→surface pass carrying ``exq`` (falling rain mass, kg/m^2). All
    arrays are ``(..., K)`` with the level axis last; the scan runs over levels
    (moved to the front internally). Returns updated ``(tdel, qdel)``.
    """
    # Move level axis (last) to front so lax.scan iterates top → surface.
    pm, ti, qi, qs, dq, td, qd = (
        jnp.moveaxis(a, -1, 0)
        for a in (pmass, tin, qin, qsat, dqsat, tdel, qdel)
    )

    def body(exq, level):
        pm_k, ti_k, qi_k, qs_k, dq_k, td_k, qd_k = level
        # Accumulate condensate falling out of condensing layers (qdel < 0).
        exq = exq + jnp.where(qd_k < 0.0, -qd_k * pm_k, 0.0)
        # Re-evaporate where this layer is not condensing but rain falls through it.
        evap = (qd_k >= 0.0) & (exq > 0.0)
        deficit = (qs_k - qi_k) / (1.0 + hlcp * dq_k)          # saturation deficit
        # Limited by the available rain expressed in specific-humidity units.
        deficit = jnp.minimum(jnp.maximum(deficit, 0.0), exq / pm_k)
        deff = jnp.where(evap, deficit, 0.0)
        qd_k = qd_k + deff
        td_k = td_k - deff * hlcp
        exq = exq - deff * pm_k
        return exq, (td_k, qd_k)

    exq0 = jnp.zeros(pmass.shape[:-1])
    _, (td_out, qd_out) = jax.lax.scan(body, exq0, (pm, ti, qi, qs, dq, td, qd))
    return jnp.moveaxis(td_out, 0, -1), jnp.moveaxis(qd_out, 0, -1)


def lscale_cond(
    tin: Array,
    qin: Array,
    p_full: Array,
    p_half: Array,
    hc: float = 1.0,
    do_evap: bool = True,
) -> tuple[Array, Array, Array]:
    """Large-scale condensation (``do_simple``, no snow).

    Args:
        tin:    temperature at full levels ``(..., K)`` [K].
        qin:    specific humidity at full levels ``(..., K)`` [kg/kg].
        p_full: pressure at full levels ``(..., K)`` [Pa].
        p_half: pressure at half levels ``(..., K+1)`` [Pa].
        hc:     relative-humidity condensation threshold (Frierson default 1.0).
        do_evap: re-evaporate falling rain in sub-saturated layers (Frierson True).

    Returns:
        ``(rain, tdel, qdel)`` — column-integrated rain ``(...,)`` [kg/m^2], and
        the temperature/humidity **increments** ``(..., K)`` (not rates; divide
        by the physics timestep as the driver does).
    """
    hlcp = constants.HLV / constants.CP_AIR                    # do_simple (F90 L134)
    qsat, dqsat = saturation_specific_humidity_and_deriv(tin, p_full, hc=hc)

    # Adjust only supersaturated layers: (qin - qsat)*qsat > 0 (F90 L155).
    do_adjust = (qin - qsat) * qsat > 0.0
    qdel = jnp.where(do_adjust, (qsat - qin) / (1.0 + hlcp * dqsat), 0.0)
    tdel = jnp.where(do_adjust, -hlcp * qdel, 0.0)

    # Pressure mass of each layer (F90 L175).
    pmass = (p_half[..., 1:] - p_half[..., :-1]) / constants.GRAV

    if do_evap:
        tdel, qdel = _precip_evap(pmass, tin, qin, qsat, dqsat, hlcp, tdel, qdel)

    # Column-integrated precipitation, floored at zero (F90 L190-194).
    precip = jnp.maximum(-jnp.sum(pmass * qdel, axis=-1), 0.0)
    rain = precip                                             # do_simple: no snow
    return rain, tdel, qdel
