"""Saturation vapor pressure and specific humidity — Isca's ``do_simple`` path.

Faithful port of the ``do_simple`` branch of Isca's
``src/shared/sat_vapor_pres/sat_vapor_pres_k.F90``. Frierson's moist aquaplanet
(and the rest of Isca's idealized moist physics) runs with
``sat_vapor_pres_nml: do_simple = .true.``, which uses a constant-latent-heat
Clausius-Clapeyron saturation vapor pressure rather than the Smithsonian lookup
tables.

Saturation vapor pressure over liquid (``sat_vapor_pres_k.F90`` L233-239, the
``do_simple`` table construction)::

    es(T)     = ES0 * 610.78 * exp(-HLV/RVGAS * (1/T - 1/TFREEZE))
    des/dT(T) = HLV * es / (RVGAS * T^2)

with ``ES0 = 1`` (the default namelist scaling). Saturation specific humidity
(``compute_qs_k`` L760-776). The **default** branch
(``use_exact_qs = .false.``, which Frierson uses) accounts for the vapor's own
contribution to the total pressure::

    denom  = p - (1 - eps) * es                      eps = RDGAS / RVGAS
    qs     = eps * es / denom        (= eps if denom <= 0)
    dqs/dT = eps * p * (des/dT) / denom^2

The ``use_exact_qs = .true.`` branch (misleadingly named — it is the *linear*
approximation) gives ``qs = (1 + zvir*q)*eps*es/p``; that path is not used by
Frierson and is offered here as an option.

**Deviation from Isca (documented):** Isca evaluates ``es`` by quadratic
interpolation of a precomputed table (``lookup_es``); jsca evaluates the closed
form directly. The table is *built* from exactly this formula, so the only
difference is the table's interpolation error, which the fixture test bounds
(``tests/test_sat_vapor_pres_fixtures.py``). Direct evaluation is both simpler
and marginally more accurate, and is natural under JAX (no gather/interpolation).
"""
from __future__ import annotations

import jax.numpy as jnp

from jsca import constants

Array = jnp.ndarray

#: ``ES0`` — the reference-pressure scaling in ``sat_vapor_pres_nml`` (default 1).
ES0 = 1.0
#: ``eps = R_dry / R_vapor`` (``= WTMH2O/WTMAIR``), the molecular weight ratio.
EPS = constants.RDGAS / constants.RVGAS


def saturation_vapor_pressure(t: Array) -> Array:
    """Saturation vapor pressure ``es`` (Pa) over liquid, ``do_simple`` formula.

    Port of ``sat_vapor_pres_k.F90`` L237. ``t`` in kelvin.
    """
    return ES0 * 610.78 * jnp.exp(-constants.HLV / constants.RVGAS
                                  * (1.0 / t - 1.0 / constants.TFREEZE))


def d_saturation_vapor_pressure_dt(t: Array, es: Array | None = None) -> Array:
    """``des/dT`` (Pa/K), ``do_simple`` formula (``sat_vapor_pres_k.F90`` L238)."""
    if es is None:
        es = saturation_vapor_pressure(t)
    return constants.HLV * es / (constants.RVGAS * t**2)


def saturation_specific_humidity(t: Array, p: Array) -> Array:
    """Saturation specific humidity ``qs`` (kg/kg) — the default (Frierson) form.

    Port of ``compute_qs_k`` L767-772: ``qs = eps*es/(p - (1-eps)*es)``, falling
    back to ``eps`` where ``denom <= 0``. ``t`` in kelvin, ``p`` in Pa.
    """
    es = saturation_vapor_pressure(t)
    denom = p - (1.0 - EPS) * es
    return jnp.where(denom > 0.0, EPS * es / jnp.where(denom > 0.0, denom, 1.0), EPS)


def saturation_specific_humidity_and_deriv(t: Array, p: Array) -> tuple[Array, Array]:
    """``(qs, dqs/dT)`` together (``compute_qs_k`` L767-775).

    ``dqs/dT = eps*p*(des/dT)/denom^2`` — Isca computes this with ``denom^2``
    regardless of the sign of ``denom``.
    """
    es = saturation_vapor_pressure(t)
    des = d_saturation_vapor_pressure_dt(t, es)
    denom = p - (1.0 - EPS) * es
    safe = jnp.where(denom != 0.0, denom, 1.0)
    qs = jnp.where(denom > 0.0, EPS * es / safe, EPS)
    dqsdt = EPS * p * des / safe**2
    return qs, dqsdt
