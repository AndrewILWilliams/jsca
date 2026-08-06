"""Simplified Betts-Miller (Frierson quasi-equilibrium) convection — CAPE stage.

Faithful port of the parcel-ascent / CAPE calculation in Isca's
``src/atmos_param/qe_moist_convection/qe_moist_convection.F90`` (the Frierson
2007 "SIMPLE_BETTS_MILLER" scheme, the convection option the Frierson aquaplanet
uses). This module ports the *diagnosis* half of the scheme —
``CAPE_calculation`` and everything it calls (the LCL lookup table, the dry
ascent below the LCL, and the moist-adiabatic ascent above it), producing CAPE,
CIN, and the levels of the LCL and of zero buoyancy (LZB). The Betts-Miller
*adjustment* (reference profiles, Pq/Pt, deep/shallow relaxation → temperature
and humidity tendencies) is a separate follow-up module that consumes these.

**Namelist (Frierson, ``qe_moist_convection_nml``):** ``rhbm=0.7``, ``Tmin=160``,
``Tmax=350`` (``tau_bm`` default 7200). ``Tmin`` is the parcel-too-cold cutoff
that aborts the ascent; ``Tmin``/``Tmax`` also bound the LCL lookup table.

**Algorithm — per column (F90 ``CAPE_calculation`` L395, ``CAPE_below_LCL``
L462, ``CAPE_above_LCL`` L597):**

1. Lift the lowest-level parcel. If already saturated, wring out moisture and
   warm it (F90 L499-502); the LCL is the surface.
2. Otherwise dry-adiabatic ascent conserving potential temperature; the LCL
   temperature solves ``value = log(es/T^(1/kappa))`` (F90 ``lcl_temp``), looked
   up from a table generated once by Newton iteration (F90 ``generate_lcl_table``
   / ``get_lcl_temp``). Below the LCL the parcel accumulates CIN.
3. At and above the LCL, second-order (RK2 in ln p) moist-adiabatic ascent
   (F90 L544-563, L624-643); each level adds to CIN while sub-buoyant or to CAPE
   once buoyant, using **virtual** temperature (O'Gorman & Schneider 2008). The
   ascent stops at the level of zero buoyancy (first sub-buoyant level after CAPE
   has appeared) or where the parcel falls below ``Tmin``.

**Deviation (documented):** Isca's ``escomp`` interpolates the ``do_simple`` es
lookup table; jsca evaluates the closed form the table is built from (see
``sat_vapor_pres``). That ~1e-7 relative es difference is the only inexactness;
the CAPE arithmetic is otherwise exact. The LCL table here is likewise generated
from the closed form (NumPy, at import — an init-time artifact, not on the step
path), reproducing Isca's table-generation Newton solve.

**Index convention:** jsca returns ``kLZB``/``kLCL`` as 0-based level indices
(k=0 top … K-1 surface), pointing at the same physical level as Isca's 1-based
indices (add 1 to recover the Fortran value). ``kLZB = 0`` means "no CAPE / no
LZB", matching Isca's ``kLZB = 0`` sentinel.

Layout: column physics, **level axis last**; ``Tin``/``qin``/``p_full`` are
``(..., K)``, ``p_half`` is ``(..., K+1)``. Leading axes are batched (vmapped).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from jsca import constants

Array = jnp.ndarray

# --- scheme constants (Frierson qe_moist_convection_nml + Isca parameters) ---
_KAPPA = constants.KAPPA
_HLV = constants.HLV
_CP = constants.CP_AIR
_RDGAS = constants.RDGAS
_RVGAS = constants.RVGAS
_GRAV = constants.GRAV
_EPS = _RDGAS / _RVGAS
_PREF = 1.0e5           # F90 pref
_SMALL = 1.0e-10        # F90 small (avoids /0 in the dry limit)
TMIN = 160.0            # Frierson Tmin: parcel-too-cold cutoff + table lower bound
TMAX = 350.0            # Frierson Tmax: table upper bound
VAL_INC = 0.01          # F90 val_inc: LCL table spacing
RHBM = 0.7              # Frierson rhbm: reference relative humidity
TAU_BM = 7200.0         # Betts-Miller relaxation timescale (s), qe default


def _es_closed(t):
    """do_simple saturation vapor pressure (Isca ``escomp`` with do_simple)."""
    return 610.78 * jnp.exp(-_HLV / _RVGAS * (1.0 / t - 1.0 / constants.TFREEZE))


def _mixing_ratio(e, p):
    """F90 ``mixing_ratio``: r = eps * e / (p - e)."""
    return _RDGAS * e / _RVGAS / (p - e)


def _virtual_temp(t, r):
    """F90 ``virtual_temp`` (via specific humidity q = r/(1+r))."""
    q = r / (1.0 + r)
    return t * (1.0 + q * (_RVGAS / _RDGAS - 1.0))


# --------------------------------------------------------------------------
# LCL lookup table — generated once at import from the closed-form es, exactly
# as Isca's generate_lcl_table does by Newton iteration (init-time, NumPy).
# --------------------------------------------------------------------------
def _build_lcl_table():
    def es(T):
        return 610.78 * np.exp(-_HLV / _RVGAS * (1.0 / T - 1.0 / constants.TFREEZE))

    def val_diff(T, value):
        return value - np.log(es(T) * T ** (-1.0 / _KAPPA))

    def dval_diff(T):
        return 1.0 / _KAPPA * T ** (-1.0) - _HLV / _RVGAS * T ** (-2.0)

    def lcl_temp(value, guess):
        # Newton iteration (F90 lcl_temp): fixed convergence criterion.
        T = guess
        dT = 1e-7 + 1.0
        it = 0
        while abs(dT) > 1e-7 and it < 100:
            dT = val_diff(T, value) / dval_diff(T)
            T = T - dT
            it += 1
        return T

    val_min = np.log(es(TMIN) / (TMIN ** (1.0 / _KAPPA)))
    val_max = np.log(es(TMAX) / (TMAX ** (1.0 / _KAPPA)))
    n = int(np.ceil((val_max - val_min) / VAL_INC))
    table = np.empty(n, dtype=np.float64)
    guess = TMIN
    for k in range(n):
        table[k] = lcl_temp(val_min + k * VAL_INC, guess)
        guess = table[k]
    return jnp.asarray(table), float(val_min), float(val_max)


_LCL_TABLE, VAL_MIN, VAL_MAX = _build_lcl_table()


def _get_lcl_temp(value):
    """F90 ``get_lcl_temp``: linear interpolation of the LCL table.

    Reproduces the Fortran's real-valued floor index arithmetic. ``iv_floor`` is
    Isca's 1-based floor index; here it indexes the 0-based table as
    ``table[iv_floor]`` / ``table[iv_floor-1]``.
    """
    iv_floor = jnp.floor((value - VAL_MIN) / VAL_INC).astype(jnp.int32) + 1
    w_floor = VAL_MIN + (iv_floor - 1) * VAL_INC
    w_ceil = (value - w_floor) / VAL_INC
    hi = jnp.clip(iv_floor, 0, _LCL_TABLE.shape[0] - 1)
    lo = jnp.clip(iv_floor - 1, 0, _LCL_TABLE.shape[0] - 1)
    return _LCL_TABLE[hi] * w_ceil - _LCL_TABLE[lo] * (w_ceil - 1.0)


def _cape_column(p_full, p_half, Tin, rin):
    """Parcel ascent for one column. Returns (CAPE, CIN, kLZB, kLCL, Tp, rp).

    ``kLZB``/``kLCL`` are 0-based; ``kLZB=0`` is the "no CAPE" sentinel. ``Tp``/
    ``rp`` are the parcel temperature/mixing-ratio profiles (needed by the
    adjustment stage). Faithful to F90 ``CAPE_below_LCL`` + ``CAPE_above_LCL``.
    """
    K = Tin.shape[0]
    ks = K - 1  # surface index
    Tin_v = _virtual_temp(Tin, rin)

    T0 = Tin[ks]
    r0 = rin[ks]
    es0 = _es_closed(T0)
    rs = _mixing_ratio(es0, p_full[ks])
    saturated = r0 >= rs

    # ---- surface-saturated branch: LCL at surface, wring out moisture ----
    def sat_branch(_):
        Tp = Tin
        rp = rin
        Tp_ks = T0 + (r0 - rs) / ((_CP / (_HLV + _SMALL)) + (_HLV * rs) / _RVGAS / T0 ** 2)
        es_ks = _es_closed(Tp_ks)
        rp_ks = _mixing_ratio(es_ks, p_full[ks])
        Tp = Tp.at[ks].set(Tp_ks)
        rp = rp.at[ks].set(rp_ks)
        # (CIN, CAPE, nocape, kLFC, kLZB, kLCL, skip)
        return Tp, rp, 0.0, 0.0, True, 0, 0, ks, False

    # ---- unsaturated branch: dry ascent to LCL, then RK2 at LCL ----
    def unsat_branch(_):
        theta0 = Tin[ks] * (_PREF / p_full[ks]) ** _KAPPA

        # r0 <= 0: LCL is model top, skip moist ascent
        def r0_nonpos(_):
            return Tin, rin, 0.0, 0.0, True, 0, 0, 0, True

        def r0_pos(_):
            value = jnp.log(theta0 ** (-1.0 / _KAPPA) * _PREF * r0 / (_EPS + r0))
            TLCL0 = _get_lcl_temp(value)
            pLCL0 = _PREF * (TLCL0 / theta0) ** (1.0 / _KAPPA)
            # clamp LCL into the model domain (F90 L519-523)
            above_top = pLCL0 < p_full[0]
            pLCL = jnp.where(above_top, p_full[0], pLCL0)
            TLCL = jnp.where(above_top, theta0 * (pLCL / _PREF) ** _KAPPA, TLCL0)

            # dry-adiabatic upward integration while p_full[k] > pLCL (F90 L528-539)
            def cond(state):
                k, _CIN, _Tp, _rp = state
                return (k >= 0) & (p_full[k] > pLCL)

            def body(state):
                k, CIN, Tp, rp = state
                Tpk = theta0 * (p_full[k] / _PREF) ** _KAPPA
                rpk = _mixing_ratio(_es_closed(Tpk), p_full[k])
                CIN = CIN + _RDGAS * (Tin_v[k] - _virtual_temp(Tpk, r0)) \
                    * jnp.log(p_half[k + 1] / p_half[k])
                Tp = Tp.at[k].set(Tpk)
                rp = rp.at[k].set(rpk)
                return k - 1, CIN, Tp, rp

            k_end, CIN, Tp, rp = jax.lax.while_loop(cond, body, (ks, 0.0, Tin, rin))
            kLCL = k_end  # loop exits at the LCL level (first p_full <= pLCL)

            # RK2 to the LCL level (F90 L544-563)
            a = _KAPPA * TLCL + (_HLV / _CP) * r0
            b = (_HLV ** 2) * r0 / (_CP * _RVGAS * TLCL ** 2)
            dtdlnp = a / (1.0 + b)
            Tp_half = TLCL + dtdlnp * jnp.log(p_full[kLCL] / pLCL) / 2.0

            def too_cold(_):
                # set_values_if_nocape: abort, no CAPE
                return Tin, rin, 0.0, 0.0, True, 0, 0, 0, True

            def proceed(_):
                # second RK2 stage
                rp1 = _mixing_ratio(_es_closed(Tp_half), (p_full[kLCL] + pLCL) / 2.0)
                a2 = _KAPPA * Tp_half + (_HLV / _CP) * rp1
                b2 = (_HLV ** 2) * rp1 / (_CP * _RVGAS * Tp_half ** 2)
                dtdlnp2 = a2 / (1.0 + b2)
                Tp_full = TLCL + dtdlnp2 * jnp.log(p_full[kLCL] / pLCL)

                def too_cold2(_):
                    return Tin, rin, 0.0, 0.0, True, 0, 0, 0, True

                def buoyancy(_):
                    rp2 = _mixing_ratio(_es_closed(Tp_full), p_full[kLCL])
                    Tp_l = Tp.at[kLCL].set(Tp_full)
                    rp_l = rp.at[kLCL].set(rp2)
                    tv_p = _virtual_temp(Tp_full, rp2)
                    subbuoyant = tv_p < Tin_v[kLCL]
                    dCIN = _RDGAS * (Tin_v[kLCL] - tv_p) * jnp.log(p_half[kLCL + 1] / p_half[kLCL])
                    dCAPE = _RDGAS * (tv_p - Tin_v[kLCL]) * jnp.log(p_half[kLCL + 1] / p_half[kLCL])
                    CIN2 = CIN + jnp.where(subbuoyant, dCIN, 0.0)
                    CAPE2 = jnp.where(subbuoyant, 0.0, dCAPE)
                    nocape2 = subbuoyant
                    kLFC2 = jnp.where(subbuoyant, 0, kLCL)
                    return Tp_l, rp_l, CIN2, CAPE2, nocape2, kLFC2, 0, kLCL, False

                return jax.lax.cond(Tp_full < TMIN, too_cold2, buoyancy, None)

            return jax.lax.cond(Tp_half < TMIN, too_cold, proceed, None)

        return jax.lax.cond(r0 <= 0.0, r0_nonpos, r0_pos, None)

    Tp, rp, CIN, CAPE, nocape, kLFC, kLZB, kLCL, skip = jax.lax.cond(
        saturated, sat_branch, unsat_branch, None
    )

    # ---- CAPE_above_LCL: moist ascent from kLCL-1 up to top (F90 L597) ----
    # A downward-in-k scan over all levels; only levels k < kLCL act, and a
    # `done` flag freezes state after an early exit (Fortran `go to 20`).
    def above_body(carry, k):
        Tp, rp, CIN, CAPE, nocape, kLFC, kLZB, done = carry
        active = (k < kLCL) & (~skip) & (~done)

        a1 = _KAPPA * Tp[k + 1] + (_HLV / _CP) * rp[k + 1]
        b1 = (_HLV ** 2) * rp[k + 1] / (_CP * _RVGAS * Tp[k + 1] ** 2)
        dt1 = a1 / (1.0 + b1)
        Tp_half = Tp[k + 1] + dt1 * jnp.log(p_full[k] / p_full[k + 1]) / 2.0

        cold1 = Tp_half < TMIN
        rp_half = _mixing_ratio(_es_closed(Tp_half), (p_full[k] + p_full[k + 1]) / 2.0)
        a2 = _KAPPA * Tp_half + (_HLV / _CP) * rp_half
        b2 = (_HLV ** 2) * rp_half / (_CP * _RVGAS * Tp_half ** 2)
        dt2 = a2 / (1.0 + b2)
        Tp_full = Tp[k + 1] + dt2 * jnp.log(p_full[k] / p_full[k + 1])
        cold2 = Tp_full < TMIN

        rp_full = _mixing_ratio(_es_closed(Tp_full), p_full[k])
        tv_p = _virtual_temp(Tp_full, rp_full)
        subbuoyant = tv_p < Tin_v[k]
        dCIN = _RDGAS * (Tin_v[k] - tv_p) * jnp.log(p_half[k + 1] / p_half[k])
        dCAPE = _RDGAS * (tv_p - Tin_v[k]) * jnp.log(p_half[k + 1] / p_half[k])

        # classify this level's outcome (only meaningful when `active`)
        cold = (cold1 | cold2) & nocape                      # parcel too cold -> abort
        hit_lzb = subbuoyant & (~nocape)                     # sub-buoyant after CAPE -> LZB
        add_cin = subbuoyant & nocape & (~cold1) & (~cold2)  # still sub-buoyant, no CAPE yet
        add_cape = (~subbuoyant) & (~cold1) & (~cold2)       # buoyant -> CAPE

        # apply, gated by `active`
        set_prof = active & (~cold) & (~hit_lzb)             # levels that update Tp/rp
        Tp = Tp.at[k].set(jnp.where(set_prof, Tp_full, Tp[k]))
        rp = rp.at[k].set(jnp.where(set_prof, rp_full, rp[k]))
        # cold-abort resets the whole parcel to the environment (set_values_if_nocape)
        Tp = jnp.where(active & cold, Tin, Tp)
        rp = jnp.where(active & cold, rin, rp)

        CIN = jnp.where(active & cold, 0.0, CIN + jnp.where(active & add_cin, dCIN, 0.0))
        CAPE = CAPE + jnp.where(active & add_cape, dCAPE, 0.0)
        kLZB = jnp.where(active & cold, 0, kLZB)
        kLZB = jnp.where(active & hit_lzb, k + 1, kLZB)
        kLFC = jnp.where(active & cold, 0, kLFC)
        kLFC = jnp.where(active & add_cape & nocape, k, kLFC)
        nocape = jnp.where(active & add_cape, False, nocape)
        done = done | (active & (cold | hit_lzb))

        return (Tp, rp, CIN, CAPE, nocape, kLFC, kLZB, done), None

    ks_levels = jnp.arange(K - 2, -1, -1)  # kLCL-1 .. 0, masked by `active`
    (Tp, rp, CIN, CAPE, nocape, kLFC, kLZB, done), _ = jax.lax.scan(
        above_body, (Tp, rp, CIN, CAPE, nocape, kLFC, kLZB, False), ks_levels
    )

    return CAPE, CIN, kLZB, kLCL, Tp, rp


def convective_cape(Tin: Array, qin: Array, p_full: Array, p_half: Array):
    """CAPE, CIN and LCL/LZB levels for the Frierson QE convection scheme.

    Args:
        Tin, qin, p_full: ``(..., K)`` temperature [K], specific humidity [kg/kg],
            full-level pressure [Pa].
        p_half: ``(..., K+1)`` half-level pressure [Pa].

    Returns:
        ``(CAPE, CIN, kLZB, kLCL)`` — ``CAPE``/``CIN`` ``(...)`` [J/kg];
        ``kLZB``/``kLCL`` ``(...)`` 0-based level indices (``kLZB=0`` = no CAPE;
        add 1 to recover Isca's 1-based index).
    """
    rin = qin / (1.0 - qin)  # F90: rin = qin/(1-qin)
    flat_shape = Tin.shape[:-1]
    K = Tin.shape[-1]
    tin2 = Tin.reshape(-1, K)
    rin2 = rin.reshape(-1, K)
    pf2 = p_full.reshape(-1, K)
    ph2 = p_half.reshape(-1, K + 1)

    cape, cin, klzb, klcl, _, _ = jax.vmap(_cape_column)(pf2, ph2, tin2, rin2)
    return (cape.reshape(flat_shape), cin.reshape(flat_shape),
            klzb.reshape(flat_shape), klcl.reshape(flat_shape))


# ==========================================================================
# Stage 3b — the Betts-Miller adjustment (F90 SBM_convection_scheme L327-389
# and its helpers). Given the parcel ascent, relaxes T and q toward reference
# profiles over the convective layer [kLZB, k_surface], with the Frierson
# "shallower" shallow-convection scheme and enthalpy-conserving deep adjustment.
# ==========================================================================
def _adjust_column(p_full, p_half, Tin, qin, Tp, rp, kLZB, CAPE, dt):
    """Betts-Miller adjustment for one column. Faithful to F90
    set_reference_profiles + Pq/Pt + do_deep_convection + do_shallow_convection.

    Returns ``(deltaT, deltaq, rain, convflag, Tref, qref)``. ``deltaT``/
    ``deltaq`` are increments over the step (the driver divides by ``dt``);
    ``rain`` is column-integrated (kg/m^2); ``convflag`` is 0 (no CAPE),
    1 (shallow / inactive) or 2 (deep).
    """
    K = Tin.shape[0]
    ks = K - 1
    kk = jnp.arange(K)
    dp_low = p_half[:-1] - p_half[1:]   # p_half[k]   - p_half[k+1]  (< 0)
    dp_up = p_half[1:] - p_half[:-1]    # p_half[k+1] - p_half[k]    (> 0)
    below = kk >= kLZB                  # convective layer [kLZB, ks]

    def no_conv(_):
        # CAPE <= 0: everything set to full model values (F90 L370-372).
        z = jnp.zeros(K)
        return z, z, 0.0, 0, Tin, qin

    def with_cape(_):
        # --- set_reference_profiles (F90 L768) ---
        Tref = Tp
        eref = RHBM * p_full * rp / (rp + _EPS)
        rp_ref = _mixing_ratio(eref, p_full)
        qref_ref = rp_ref / (1.0 + rp_ref)
        qref = jnp.where(below, qref_ref, qin)
        # zero out above the LZB: levels [0, max(kLZB,1)-1] -> environment
        kk_top = jnp.maximum(kLZB, 1) - 1
        above = kk <= kk_top
        Tref = jnp.where(above, Tin, Tref)
        qref = jnp.where(above, qin, qref)

        # --- Pq / Pt (F90 L715, L741) ---
        deltaq = jnp.where(below, -(qin - qref) * dt / TAU_BM, 0.0)
        Pq = jnp.sum(jnp.where(below, deltaq * dp_low, 0.0)) / _GRAV
        deltaT = jnp.where(below, -(Tin - Tref) * dt / TAU_BM, 0.0)
        Pt = jnp.sum(jnp.where(below, (_CP / (_HLV + _SMALL)) * deltaT * dp_up, 0.0)) / _GRAV

        # ---- deep convection (Pq>0 and Pt>0), F90 do_deep_convection ----
        def deep(_):
            def timescale(_):
                # Pq > Pt: rescale deltaq, cap precip at Pt (F90 L995).
                invtau_q = Pt / Pq / TAU_BM
                dq = jnp.where(below, TAU_BM * invtau_q * deltaq, deltaq)
                return deltaT, dq, Pt, Tref
            def tref_shift(_):
                # Pq <= Pt: enthalpy-conserving uniform Tref shift (F90 L960);
                # deltaq is unchanged, precip stays Pq.
                deltak = -jnp.sum(jnp.where(below, (deltaT + (_HLV / _CP) * deltaq) * dp_up, 0.0)) \
                    / (p_half[ks + 1] - p_half[kLZB])
                dT = jnp.where(below, deltaT + deltak, deltaT)
                Tr = jnp.where(below, Tref + deltak * TAU_BM / dt, Tref)
                return dT, deltaq, Pq, Tr
            dT, dq, rain, Tr = jax.lax.cond(Pq > Pt, timescale, tref_shift, None)
            return dT, dq, rain, 2, Tr, qref

        # ---- shallow convection (Pt>0, Pq<=0), F90 do_shallow_convection ----
        def shallow(_):
            # level_of_zero_precip: integrate precip downward until it turns
            # positive (F90 L845). Carry (k, Pql).
            def cond(state):
                k, Pql = state
                return (Pql < 0.0) & (k <= ks)

            def body(state):
                k, Pql = state
                Pql = Pql - deltaq[k] * dp_low[k] / _GRAV
                return k + 1, Pql

            k_end, Pql = jax.lax.while_loop(cond, body, (kLZB, Pq))
            k_top = k_end - 1
            found = Pql > 0.0

            # above the zero-precip level: reset to environment (F90 L882)
            zmask = (kk >= kLZB) & (kk <= k_top - 1) & (k_top > kLZB)
            Tref_s = jnp.where(zmask, Tin, Tref)
            qref_s = jnp.where(zmask, qin, qref)
            dT_s = jnp.where(zmask, 0.0, deltaT)
            dq_s = jnp.where(zmask, 0.0, deltaq)

            def found_branch(_):
                # change_Tref_LZB_shallowconv (F90 L892): scale the top layer so
                # precip is exactly zero, then shift Tref to conserve enthalpy.
                c = Pql * _GRAV / (dq_s[k_top] * dp_up[k_top])
                dq2 = jnp.where(kk == k_top, dq_s * c, dq_s)
                dT2 = jnp.where(kk == k_top, dT_s * c, dT_s)
                topmask = kk >= k_top   # [k_top, ks]
                deltak = jnp.sum(jnp.where(topmask, dT2 * dp_low, 0.0)) \
                    / (p_half[ks + 1] - p_half[k_top])
                apply = topmask & (k_top != ks)
                dT3 = jnp.where(apply, dT2 + deltak, dT2)
                Tr3 = jnp.where(apply, Tref_s + deltak * TAU_BM / dt, Tref_s)
                return dT3, dq2, Tr3, qref_s

            def notfound_branch(_):
                # no zero-precip level: reset the (rest of the) convective layer.
                # k_top==kLZB -> just the surface; else [kLZB, k_top].
                m_surf = kk == ks
                m_layer = (kk >= kLZB) & (kk <= k_top)
                m = jnp.where(k_top == kLZB, m_surf, m_layer)
                Tr = jnp.where(m, Tin, Tref_s)
                qr = jnp.where(m, qin, qref_s)
                dT = jnp.where(m, 0.0, dT_s)
                dq = jnp.where(m, 0.0, dq_s)
                return dT, dq, Tr, qr

            dT, dq, Tr, qr = jax.lax.cond(found, found_branch, notfound_branch, None)
            return dT, dq, 0.0, 1, Tr, qr   # shallow: no precip

        def deep_or_shallow(_):
            return jax.lax.cond(Pt > 0.0, shallow, inactive, None)

        def inactive(_):
            # CAPE>0 but Pt<=0: no adjustment, convflag stays 1 (F90 L360-364).
            z = jnp.zeros(K)
            return z, z, 0.0, 1, Tin, qin

        return jax.lax.cond((Pq > 0.0) & (Pt > 0.0), deep, deep_or_shallow, None)

    return jax.lax.cond(CAPE > 0.0, with_cape, no_conv, None)


def qe_moist_convection(Tin: Array, qin: Array, p_full: Array, p_half: Array,
                        dt: float):
    """Frierson simplified Betts-Miller convection (full scheme).

    Faithful port of ``qe_moist_convection`` (``SIMPLE_BETTS_MILLER``): the
    parcel-ascent CAPE diagnosis (stage 3a) followed by the Betts-Miller
    adjustment (stage 3b). Relaxes temperature and humidity toward post-ascent
    reference profiles over the convective layer.

    Args:
        Tin, qin, p_full: ``(..., K)`` temperature [K], specific humidity
            [kg/kg], full-level pressure [Pa].
        p_half: ``(..., K+1)`` half-level pressure [Pa].
        dt: physics timestep [s] (Betts-Miller uses ``tau_bm = 7200`` s).

    Returns:
        ``(rain, deltaT, deltaq, convflag)`` — column-integrated convective rain
        ``(...)`` [kg/m^2]; the temperature/humidity **increments** ``(..., K)``
        over the step (the Isca driver divides these by ``dt`` to get rates);
        and ``convflag`` ``(...)`` (0 none / 1 shallow / 2 deep).

    Note: the internal reference profiles ``Tref``/``qref`` are diagnostics not
    returned here. ``qref`` in particular carries an irreducible knife-edge: for
    a shallow column whose convective layer is a net moisture *sink*, the
    "level of zero precipitation" test compares a quantity that is exactly zero
    in exact arithmetic (``Pq - Pq``), so the ~1e-7 ``sat_vapor_pres`` es
    deviation can flip the discrete ``found`` branch and change ``qref`` at the
    surface. The returned tendencies are unaffected (both branches coincide
    there), which is why ``deltaT``/``deltaq``/``rain`` match Isca to ~1e-7.
    """
    rin = qin / (1.0 - qin)
    flat = Tin.shape[:-1]
    K = Tin.shape[-1]
    tin2 = Tin.reshape(-1, K)
    qin2 = qin.reshape(-1, K)
    rin2 = rin.reshape(-1, K)
    pf2 = p_full.reshape(-1, K)
    ph2 = p_half.reshape(-1, K + 1)

    def one(pf, ph, t, q, r):
        cape, _cin, klzb, _klcl, Tp, rp = _cape_column(pf, ph, t, r)
        dT, dq, rain, cflag, _Tref, _qref = _adjust_column(
            pf, ph, t, q, Tp, rp, klzb, cape, dt)
        return rain, dT, dq, cflag

    rain, dT, dq, cflag = jax.vmap(one)(pf2, ph2, tin2, qin2, rin2)
    return (rain.reshape(flat), dT.reshape(flat + (K,)),
            dq.reshape(flat + (K,)), cflag.reshape(flat))
