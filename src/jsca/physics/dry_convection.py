"""Simple dry convective adjustment — Isca's ``dry_convection`` (``DRY``).

Faithful port of ``src/atmos_param/dry_convection/dry_convection.f90`` (Schneider &
Walker 2006, via Tapio Schneider's FMS; added to the moist stack by James Penn).
It relaxes the temperature profile toward a prescribed lapse rate ``gamma`` over a
timescale ``tau`` and touches **only temperature** — no moisture, and (in the
driver) large-scale condensation is skipped when this scheme is active
(``idealized_moist_phys.F90`` L1009).

Algorithm (``capecalc`` L237-330, then the adjustment L164-203):

1. **Lift** a parcel from the lowest level upward at the prescribed lapse rate::

       zdpkpk = (p_full[k]/p_full[k+1])**(Rd/cp)
       tp[k]  = tp[k+1] + gamma*(tp[k+1]*zdpkpk - tp[k+1])

   With ``gamma = 1`` the parcel follows the dry adiabat (potential temperature).
2. **Scan upward** to find the lifting condensation level (``lcl``, cloud base),
   the level of zero buoyancy (``lzb``, cloud top), and CAPE/CIN, gating on
   "not above a lower cloud" (``lzb`` still at the surface index). Once ``lzb`` is
   set, higher levels are ambient. If ``cin > cape`` convection is switched off
   (parcel profile reset to ambient).
3. **Conserve energy**: over the convecting region ``[lzb, btm]`` shift the parcel
   profile by the mass-weighted mean temperature difference so the column-integrated
   enthalpy is unchanged; ambient elsewhere.
4. Return the tendency ``dt_tg = (tp - tg)/tau`` [K/s] (a **rate**, unlike the qe /
   Betts-Miller schemes which return increments).

Convention: column arrays are **level-last** (k = 0 top … K-1 surface), so the
Fortran ``btm = num_levels`` bottom is ``K-1`` here and "up" is decreasing ``k``.
``lzb``/``lcl`` are returned as 0-based level indices. Leading axes are batched.
No documented deviations: pure arithmetic (a power law + logs), so fixtures hit the
tight tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsca import constants

Array = jnp.ndarray


@dataclass(frozen=True)
class DryConvectionParams:
    """Static ``dry_convection_nml`` configuration (hashable; static jit arg).

    The Fortran namelist has no defaults; these mirror Isca's giant-planet test
    case (``tau = 21600`` s, ``gamma = 1.0``).
    """

    tau: float = 21600.0    # relaxation timescale [s]
    gamma: float = 1.0      # prescribed lapse rate [non-dim]; 1 = dry adiabat


def _dry_convection_column(dp: DryConvectionParams, tg, p_full, p_half, dp_half):
    """One column (all ``(K,)``, level-last). Returns ``(dt_tg, cape, cin, lzb, lcl)``."""
    K = tg.shape[0]
    btm = K - 1                                   # surface (lowest) level index
    cons1 = constants.RDGAS / constants.CP_AIR

    # --- 1. lift parcel from the surface upward (F90 L271-275) ---
    # tp[btm] = tg[btm]; tp[k] = tp[k+1] + gamma*(tp[k+1]*zdpkpk - tp[k+1]).
    # Reverse scan (surface -> top): carry tp[k+1], index k = btm-1 .. 0.
    ratio = p_full[:-1] / p_full[1:]              # p_full[k]/p_full[k+1], k = 0..K-2
    zdpkpk = ratio ** cons1

    def lift(tp_below, zk):
        tp_k = tp_below + dp.gamma * (tp_below * zk - tp_below)
        return tp_k, tp_k

    _, tp_above = jax.lax.scan(lift, tg[btm], zdpkpk, reverse=True)  # tp_above[k], k=0..K-2
    tp = jnp.concatenate([tp_above, tg[btm:btm + 1]])   # (K,), tp[btm] = tg[btm]

    # --- 2. find lcl / lzb / cape / cin, scanning upward (F90 L278-308) ---
    log_dp = jnp.log(p_half[1:] / p_half[:-1])    # log(p_half[k+1]/p_half[k]), (K,)
    # tp[k+1], tp[k-1] (lifted); guard the k=0 edge (top): tp[k-1] unused when k==0.
    tp_kp1 = jnp.concatenate([tp[1:], tp[-1:]])   # tp[k+1] (tp[K] -> pad, unused at btm)
    tg_kp1 = jnp.concatenate([tg[1:], tg[-1:]])
    tp_km1 = jnp.concatenate([tp[:1], tp[:-1]])   # tp[k-1] (pad at k=0, unused)
    tg_km1 = jnp.concatenate([tg[:1], tg[:-1]])
    levels = jnp.arange(btm - 1, -1, -1)          # k = btm-1 .. 0 (upward)

    def search(carry, k):
        lzb, lcl, cape, cin = carry
        not_cloud = lzb == btm                    # "not above a lower cloud"
        unstable = tp[k] > tg[k]
        buoy = constants.RDGAS * (tp[k] - tg[k]) * log_dp[k]
        # unstable & not above cloud: accumulate CAPE, maybe set lcl / lzb
        add_cape = unstable & not_cloud
        cape = cape + jnp.where(add_cape, buoy, 0.0)
        set_lcl = add_cape & (tp_kp1[k] < tg_kp1[k])
        lcl = jnp.where(set_lcl, k, lcl)
        at_top = k == 0
        set_lzb = add_cape & (at_top | (tp_km1[k] < tg_km1[k]))
        lzb = jnp.where(set_lzb, k, lzb)
        # stable & not above cloud & below LCL: accumulate CIN
        add_cin = (~unstable) & not_cloud & (lcl == btm)
        cin = cin - jnp.where(add_cin, buoy, 0.0)
        return (lzb, lcl, cape, cin), None

    (lzb, lcl, cape, cin), _ = jax.lax.scan(search, (btm, btm, 0.0, 0.0), levels)

    # cin > cape -> switch convection off (parcel = ambient) (F90 L312)
    off = cin > cape
    # no convection detected -> cape = cin = 0 (F90 L322-325)
    none = (lcl == btm) & (lzb == btm)
    cape = jnp.where(none, 0.0, cape)
    cin = jnp.where(none, 0.0, cin)

    # --- 3. energy-conserving adjustment over [lzb, btm] (F90 L177-200) ---
    k_idx = jnp.arange(K)
    convecting = (k_idx >= lzb) & (k_idx <= btm) & (~off)
    dp_conv = jnp.where(convecting, dp_half, 0.0)
    denom = jnp.sum(dp_conv)
    ener_int = jnp.sum(dp_conv * (tg - tp)) / jnp.where(denom != 0.0, denom, 1.0)
    tp_adj = jnp.where(convecting, tp + ener_int, tg)

    # --- 4. tendency (rate) ---
    dt_tg = (tp_adj - tg) / dp.tau
    return dt_tg, cape, cin, lzb, lcl


def dry_convection(dp: DryConvectionParams, tg: Array, p_full: Array,
                   p_half: Array):
    """Dry convective adjustment over batched columns (level-last).

    ``tg``/``p_full`` are ``(..., K)``; ``p_half`` is ``(..., K+1)``. Returns
    ``(dt_tg, cape, cin, lzb, lcl)`` — the temperature tendency ``(..., K)`` [K/s],
    CAPE/CIN ``(...)`` [J/kg], and the 0-based ``lzb``/``lcl`` level indices ``(...)``.
    """
    flat = tg.shape[:-1]
    K = tg.shape[-1]
    tg2 = tg.reshape(-1, K)
    pf2 = p_full.reshape(-1, K)
    ph2 = p_half.reshape(-1, K + 1)
    dp_half = ph2[:, 1:] - ph2[:, :-1]

    dt_tg, cape, cin, lzb, lcl = jax.vmap(
        _dry_convection_column, in_axes=(None, 0, 0, 0, 0))(dp, tg2, pf2, ph2, dp_half)
    return (dt_tg.reshape(flat + (K,)), cape.reshape(flat), cin.reshape(flat),
            lzb.reshape(flat), lcl.reshape(flat))
