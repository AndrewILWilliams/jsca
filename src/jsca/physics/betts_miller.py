"""Full Betts-Miller convection — Isca's ``betts_miller`` (``FULL_BETTS_MILLER``).

Faithful port of ``src/atmos_param/betts_miller/betts_miller.f90``, the classic
Betts-Miller (1986) convective adjustment. It is **distinct** from the simple
quasi-equilibrium scheme in :mod:`jsca.physics.qe_moist_convection`: it has its own
``capecalcnew`` parcel ascent (different LCL formula, its own hardcoded LCL lookup
table, and — unlike the qe scheme — **no virtual-temperature effects** in CAPE/CIN,
F90 L457-458) and its own reference-profile relaxation with the energy-conserving
timescale adjustment.

Only the default namelist configuration is ported (``do_simp=.true.``,
``do_shallower=.false.``, ``do_changeqref=.false.``, ``do_envsat=.false.``,
``do_taucape=.false.``, ``buoyancy_kick=0``) — what ``convection_scheme='FULL_BETTS_MILLER'``
selects. In this configuration:

1. **``capecalcnew``** (F90 L446-785): lift the surface parcel, find the LCL
   (analytic + the hardcoded ``lcltabl`` lookup), dry-adiabatic ascent below it,
   RK2 moist-adiabatic ascent above, accumulating CAPE/CIN and the level of zero
   buoyancy ``klzb``. Returns the parcel temperature ``tpc`` and mixing ratio ``rpc``.
2. **Reference profiles** (F90 L208-229): where there is CAPE, ``t_ref`` is the
   parcel temperature and ``q_ref`` is ``rhbm`` times the parcel saturation humidity.
3. **Relaxation** (F90 L249-259): ``tdel = -(t - t_ref)/tau_bm*dt``,
   ``qdel = -(q - q_ref)/tau_bm*dt`` over the convecting region ``[klzb, surface]``.
4. **Energy conservation** (``do_simp``, F90 L260-300): if the moisture and
   temperature precipitation integrals are both positive, lengthen whichever
   relaxation is "too strong" so column enthalpy is conserved; the deep-convection
   ``bmflag=2`` case. Otherwise (shallow / inconsistent) no adjustment is applied
   (the ``do_shallower``/``do_changeqref`` variants are not ported).

Convention: column arrays **level-last** (k = 0 top … K-1 surface); ``klzb``/``klcl``
returned as 0-based indices (``klzb`` = -1 when there is no CAPE). Reuses the
saturation vapour pressure ``_es_closed`` from :mod:`jsca.physics.qe_moist_convection`
(the column/Frierson configs run ``sat_vapor_pres`` with ``do_simple=.true.``, so
``escomp`` is that closed form). No documented deviations; fixtures hit the log/exp
tolerance band.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from jsca import constants
from jsca.physics.qe_moist_convection import _es_closed

Array = jnp.ndarray

_KAPPA = constants.KAPPA
_HLV = constants.HLV
_CP = constants.CP_AIR
_RDGAS = constants.RDGAS
_RVGAS = constants.RVGAS
_GRAV = constants.GRAV
_EPS = _RDGAS / _RVGAS       # rdgas/rvgas
_PSTAR = 1.0e5               # F90 pstar
_ES0 = 1.0                   # F90 es0 (DEF_ES0)
_SMALL = 1.0e-10             # F90 small
_TMIN = 173.16               # F90 parcel-too-cold cutoff (and LCL-table lower bound)

# Hardcoded LCL lookup table (F90 lcltabl `data lcltable`): 127 values, value grid
# from -23.0 to -10.4 with increment 0.1. Generated once for Isca's escomp; used
# verbatim (do NOT regenerate — it encodes the reference saturation curve).
_BM_LCL_TABLE = jnp.array([
    1.7364512e+02, 1.7427449e+02, 1.7490874e+02, 1.7554791e+02, 1.7619208e+02, 1.7684130e+02,
    1.7749563e+02, 1.7815514e+02, 1.7881989e+02, 1.7948995e+02, 1.8016539e+02, 1.8084626e+02,
    1.8153265e+02, 1.8222461e+02, 1.8292223e+02, 1.8362557e+02, 1.8433471e+02, 1.8504972e+02,
    1.8577068e+02, 1.8649767e+02, 1.8723077e+02, 1.8797006e+02, 1.8871561e+02, 1.8946752e+02,
    1.9022587e+02, 1.9099074e+02, 1.9176222e+02, 1.9254042e+02, 1.9332540e+02, 1.9411728e+02,
    1.9491614e+02, 1.9572209e+02, 1.9653521e+02, 1.9735562e+02, 1.9818341e+02, 1.9901870e+02,
    1.9986158e+02, 2.0071216e+02, 2.0157057e+02, 2.0243690e+02, 2.0331128e+02, 2.0419383e+02,
    2.0508466e+02, 2.0598391e+02, 2.0689168e+02, 2.0780812e+02, 2.0873335e+02, 2.0966751e+02,
    2.1061074e+02, 2.1156316e+02, 2.1252493e+02, 2.1349619e+02, 2.1447709e+02, 2.1546778e+02,
    2.1646842e+02, 2.1747916e+02, 2.1850016e+02, 2.1953160e+02, 2.2057364e+02, 2.2162645e+02,
    2.2269022e+02, 2.2376511e+02, 2.2485133e+02, 2.2594905e+02, 2.2705847e+02, 2.2817979e+02,
    2.2931322e+02, 2.3045895e+02, 2.3161721e+02, 2.3278821e+02, 2.3397218e+02, 2.3516935e+02,
    2.3637994e+02, 2.3760420e+02, 2.3884238e+02, 2.4009473e+02, 2.4136150e+02, 2.4264297e+02,
    2.4393941e+02, 2.4525110e+02, 2.4657831e+02, 2.4792136e+02, 2.4928053e+02, 2.5065615e+02,
    2.5204853e+02, 2.5345799e+02, 2.5488487e+02, 2.5632953e+02, 2.5779231e+02, 2.5927358e+02,
    2.6077372e+02, 2.6229310e+02, 2.6383214e+02, 2.6539124e+02, 2.6697081e+02, 2.6857130e+02,
    2.7019315e+02, 2.7183682e+02, 2.7350278e+02, 2.7519152e+02, 2.7690354e+02, 2.7863937e+02,
    2.8039954e+02, 2.8218459e+02, 2.8399511e+02, 2.8583167e+02, 2.8769489e+02, 2.8958539e+02,
    2.9150383e+02, 2.9345086e+02, 2.9542719e+02, 2.9743353e+02, 2.9947061e+02, 3.0153922e+02,
    3.0364014e+02, 3.0577420e+02, 3.0794224e+02, 3.1014515e+02, 3.1238386e+02, 3.1465930e+02,
    3.1697246e+02, 3.1932437e+02, 3.2171609e+02, 3.2414873e+02, 3.2662343e+02, 3.2914139e+02,
    3.3170385e+02,
])


@dataclass(frozen=True)
class BettsMillerParams:
    """Static ``betts_miller_nml`` configuration (hashable; static jit arg).

    Defaults mirror Isca's namelist. Only the ``do_simp=.true.`` path (with the
    shallow-convection variants off) is implemented.
    """

    tau_bm: float = 7200.0   # relaxation timescale [s]
    rhbm: float = 0.8        # target relative humidity
    do_simp: bool = True


def _bm_rmix(es, p):
    """Betts-Miller mixing ratio ``rdgas/rvgas * es/p`` (F90 e.g. L532, L522).

    Note this is ``eps*es/p``, NOT ``eps*es/(p-es)`` — the scheme drops the
    ``-es`` in the denominator throughout ``capecalcnew``.
    """
    return _EPS * es / p


def _bm_lcltabl(value):
    """LCL temperature by interpolating the hardcoded table (F90 ``lcltabl`` L788).

    ``v1 = clip(value, -23.0, -10.4); ival = floor(10*(v1+23));
    tlcl = (v2+1-10*v1)*table[ival] + (10*v1-v2)*table[ival+1]`` with
    ``v2 = -230 + ival`` (converted to 0-based indexing).
    """
    v1 = jnp.clip(value, -23.0, -10.4)
    ival = jnp.floor(10.0 * (v1 + 23.0)).astype(jnp.int32)
    v2 = -230.0 + ival
    v1s = 10.0 * v1
    lo = jnp.clip(ival, 0, _BM_LCL_TABLE.shape[0] - 1)          # F90 lcltable(ival+1) -> [ival]
    hi = jnp.clip(ival + 1, 0, _BM_LCL_TABLE.shape[0] - 1)      # F90 lcltable(ival+2) -> [ival+1]
    return (v2 + 1.0 - v1s) * _BM_LCL_TABLE[lo] + (v1s - v2) * _BM_LCL_TABLE[hi]


def _capecalc_column(p, phalf, tin, rin):
    """Parcel ascent for one column (F90 ``capecalcnew``, ``avgbl=.false.``).

    ``p``/``tin``/``rin`` are ``(K,)`` (level-last); ``phalf`` is ``(K+1,)``.
    Returns ``(cape, cin, klzb, klcl, tp, rp)`` with 0-based ``klzb`` (=-1 when no
    CAPE) / ``klcl``, and the parcel temperature/mixing-ratio profiles.
    """
    K = tin.shape[0]
    ks = K - 1

    t0 = tin[ks]
    r0 = rin[ks]
    es_sfc = _es_closed(t0)
    rs = _bm_rmix(es_sfc, p[ks])
    saturated = r0 >= rs

    def sat_branch(_):
        # already saturated at the surface: LCL at surface, wring out moisture
        tp_ks = t0 + (r0 - rs) / (_CP / (_HLV + _SMALL) + _HLV * rs / _RVGAS / t0 ** 2)
        rp_ks = _bm_rmix(_es_closed(tp_ks), p[ks])
        tp = tin.at[ks].set(tp_ks)
        rp = rin.at[ks].set(rp_ks)
        # (tp, rp, cin, cape, nocape, klfc, klzb, klcl, skip)
        return tp, rp, 0.0, 0.0, True, 0, 0, ks, False

    def unsat_branch(_):
        theta0 = t0 * (_PSTAR / p[ks]) ** _KAPPA

        def r0_nonpos(_):
            # r0 <= 0: LCL at model top, no CAPE -> reset (handled by skip)
            return tin, rin, 0.0, 0.0, True, 0, 0, 0, True

        def r0_pos(_):
            value = jnp.log(theta0 ** (-1.0 / _KAPPA) * r0 * _PSTAR / _EPS / _ES0)
            tlcl0 = _bm_lcltabl(value)
            plcl0 = _PSTAR * (tlcl0 / theta0) ** (1.0 / _KAPPA)
            above_top = plcl0 < p[0]
            plcl = jnp.where(above_top, p[0], plcl0)
            tlcl = jnp.where(above_top, theta0 * (plcl / _PSTAR) ** _KAPPA, tlcl0)

            # dry-adiabatic ascent below the LCL (F90 L569-576)
            def cond(state):
                k, _cin, _tp, _rp = state
                return (k >= 0) & (p[k] > plcl)

            def body(state):
                k, cin, tp, rp = state
                tpk = theta0 * (p[k] / _PSTAR) ** _KAPPA
                rpk = _bm_rmix(_es_closed(tpk), p[k])
                cin = cin + _RDGAS * (tin[k] - tpk) * jnp.log(phalf[k + 1] / phalf[k])
                return k - 1, cin, tp.at[k].set(tpk), rp.at[k].set(rpk)

            k_end, cin, tp, rp = jax.lax.while_loop(cond, body, (ks, 0.0, tin, rin))
            klcl = jnp.maximum(k_end, 1)        # F90 "if (klcl.eq.1) klcl = 2" (0-based: >=1)

            # RK2 saturated ascent to the LCL level (F90 L584-608)
            a1 = _KAPPA * tlcl + (_HLV / _CP) * r0
            b1 = _HLV ** 2 * r0 / _CP / _RVGAS / tlcl ** 2
            dt1 = a1 / (1.0 + b1)
            tp_half = tlcl + dt1 * jnp.log(p[klcl] / plcl) / 2.0

            def too_cold(_):
                return tin, rin, 0.0, 0.0, True, 0, 0, 0, True

            def proceed(_):
                rp_half = _bm_rmix(_es_closed(tp_half), (p[klcl] + plcl) / 2.0)
                a2 = _KAPPA * tp_half + (_HLV / _CP) * rp_half
                b2 = _HLV ** 2 / _CP / _RVGAS * rp_half / tp_half ** 2
                dt2 = a2 / (1.0 + b2)
                tp_full = tlcl + dt2 * jnp.log(p[klcl] / plcl)

                def too_cold2(_):
                    return tin, rin, 0.0, 0.0, True, 0, 0, 0, True

                def buoyant(_):
                    rp_full = _bm_rmix(_es_closed(tp_full), p[klcl])
                    tp_l = tp.at[klcl].set(tp_full)
                    rp_l = rp.at[klcl].set(rp_full)
                    sub = tp_full < tin[klcl]
                    dcin = _RDGAS * (tin[klcl] - tp_full) * jnp.log(phalf[klcl + 1] / phalf[klcl])
                    dcape = _RDGAS * (tp_full - tin[klcl]) * jnp.log(phalf[klcl + 1] / phalf[klcl])
                    cin2 = cin + jnp.where(sub, dcin, 0.0)
                    cape2 = jnp.where(sub, 0.0, dcape)
                    nocape2 = sub
                    klfc2 = jnp.where(sub, 0, klcl)
                    return tp_l, rp_l, cin2, cape2, nocape2, klfc2, 0, klcl, False

                return jax.lax.cond(tp_full < _TMIN, too_cold2, buoyant, None)

            return jax.lax.cond(tp_half < _TMIN, too_cold, proceed, None)

        return jax.lax.cond(r0 <= 0.0, r0_nonpos, r0_pos, None)

    tp, rp, cin, cape, nocape, klfc, klzb, klcl, skip = jax.lax.cond(
        saturated, sat_branch, unsat_branch, None)

    # moist ascent from klcl-1 up to the top (F90 L725-769): a downward-in-k scan;
    # only k < klcl acts, and `done` freezes state after the LZB / too-cold exit.
    def above_body(carry, k):
        tp, rp, cin, cape, nocape, klfc, klzb, done = carry
        active = (k < klcl) & (~skip) & (~done)

        a1 = _KAPPA * tp[k + 1] + (_HLV / _CP) * rp[k + 1]
        b1 = _HLV ** 2 / _CP / _RVGAS * rp[k + 1] / tp[k + 1] ** 2
        dt1 = a1 / (1.0 + b1)
        tp_half = tp[k + 1] + dt1 * jnp.log(p[k] / p[k + 1]) / 2.0
        cold1 = tp_half < _TMIN

        rp_half = _bm_rmix(_es_closed(tp_half), (p[k] + p[k + 1]) / 2.0)
        a2 = _KAPPA * tp_half + (_HLV / _CP) * rp_half
        b2 = _HLV ** 2 / _CP / _RVGAS * rp_half / tp_half ** 2
        dt2 = a2 / (1.0 + b2)
        tp_full = tp[k + 1] + dt2 * jnp.log(p[k] / p[k + 1])
        cold2 = tp_full < _TMIN

        rp_full = _bm_rmix(_es_closed(tp_full), p[k])
        sub = tp_full < tin[k]
        dcin = _RDGAS * (tin[k] - tp_full) * jnp.log(phalf[k + 1] / phalf[k])
        dcape = _RDGAS * (tp_full - tin[k]) * jnp.log(phalf[k + 1] / phalf[k])

        cold = (cold1 | cold2) & nocape          # too cold before any CAPE -> abort
        hit_lzb = sub & (~nocape)                # sub-buoyant after CAPE -> LZB, stop
        add_cin = sub & nocape & (~cold1) & (~cold2)
        add_cape = (~sub) & (~cold1) & (~cold2)

        set_prof = active & (~cold) & (~hit_lzb)
        tp = tp.at[k].set(jnp.where(set_prof, tp_full, tp[k]))
        rp = rp.at[k].set(jnp.where(set_prof, rp_full, rp[k]))
        tp = jnp.where(active & cold, tin, tp)
        rp = jnp.where(active & cold, rin, rp)

        cin = jnp.where(active & cold, 0.0, cin + jnp.where(active & add_cin, dcin, 0.0))
        cape = cape + jnp.where(active & add_cape, dcape, 0.0)
        klzb = jnp.where(active & cold, 0, klzb)
        klzb = jnp.where(active & hit_lzb, k + 1, klzb)
        klfc = jnp.where(active & cold, 0, klfc)
        klfc = jnp.where(active & add_cape & nocape, k, klfc)
        nocape = jnp.where(active & add_cape, False, nocape)
        done = done | (active & (cold | hit_lzb))
        return (tp, rp, cin, cape, nocape, klfc, klzb, done), None

    ks_levels = jnp.arange(K - 2, -1, -1)
    (tp, rp, cin, cape, nocape, klfc, klzb, done), _ = jax.lax.scan(
        above_body, (tp, rp, cin, cape, nocape, klfc, klzb, False), ks_levels)

    # nocape -> no LZB found: reset parcel to environment (F90 L770-779)
    reset = nocape
    tp = jnp.where(reset, tin, tp)
    rp = jnp.where(reset, rin, rp)
    cin = jnp.where(reset, 0.0, cin)
    klzb = jnp.where(reset, 0, klzb)
    # The scan stores klzb = k+1 with k the 0-based sub-buoyant level, which is
    # already the 0-based LZB level index (== F90 1-based klzb minus 1). 0 is the
    # "no LZB" sentinel -> return -1 there.
    klzb0 = jnp.where(klzb == 0, -1, klzb)
    return cape, cin, klzb0, klcl, tp, rp


def _betts_miller_column(bp: BettsMillerParams, dt, tin, qin, pfull, phalf):
    """One column of the Betts-Miller adjustment (do_simp default path).

    Returns ``(tdel, qdel, rain, cape, cin, klzb, klcl, t_ref, q_ref)`` — the T/q
    increments ``(K,)``, precipitation (kg/m^2), CAPE/CIN, the 0-based klzb/klcl,
    and the reference profiles.
    """
    K = tin.shape[0]
    rin = qin / (1.0 - qin)
    cape, cin, klzb, klcl, tpc, rpc = _capecalc_column(pfull, phalf, tin, rin)

    has_cape = cape > 0.0
    k_idx = jnp.arange(K)
    conv = (k_idx >= klzb) & has_cape            # convecting region [klzb, surface]

    # reference profiles (F90 L212-228, do_envsat=.false.)
    rpc_ref = bp.rhbm * rpc
    q_ref_conv = rpc_ref / (1.0 + rpc_ref)
    t_ref = jnp.where(conv, tpc, tin)
    q_ref = jnp.where(conv, q_ref_conv, qin)

    # relaxation (F90 L251-252)
    tdel = jnp.where(conv, -(tin - t_ref) / bp.tau_bm * dt, 0.0)
    qdel = jnp.where(conv, -(qin - q_ref) / bp.tau_bm * dt, 0.0)

    dp = phalf[1:] - phalf[:-1]                  # (K,)
    precip = jnp.sum(-qdel * dp / _GRAV)
    precip_t = jnp.sum(_CP / (_HLV + _SMALL) * tdel * dp / _GRAV)

    # energy conservation (do_simp, F90 L260-300)
    deep = has_cape & (precip > 0.0) & (precip_t > 0.0)
    q_heavy = precip > precip_t
    # if q precip heavier: shorten q relaxation (scale qdel), precip = precip_t
    q_scale = jnp.where(deep & q_heavy, precip_t / precip, 1.0)
    # else (do_simp): scale tdel
    t_scale = jnp.where(deep & (~q_heavy), precip / precip_t, 1.0)
    tdel = tdel * t_scale
    qdel = qdel * q_scale
    precip_final = jnp.where(deep, jnp.where(q_heavy, precip_t, precip), 0.0)

    # not deep (shallow / inconsistent): no adjustment (do_shallower/do_changeqref
    # not ported; the default `else` branch zeroes everything, F90 L406-412/414-421)
    tdel = jnp.where(deep, tdel, 0.0)
    qdel = jnp.where(deep, qdel, 0.0)
    rain = jnp.where(deep, precip_final, 0.0)
    return tdel, qdel, rain, cape, cin, klzb, klcl, t_ref, q_ref


def betts_miller(bp: BettsMillerParams, dt: float, tin: Array, qin: Array,
                 p_full: Array, p_half: Array):
    """Full Betts-Miller convection over batched columns (level-last).

    ``tin``/``qin``/``p_full`` are ``(..., K)``; ``p_half`` is ``(..., K+1)``.
    Returns ``(rain, tdel, qdel, cape, cin, klzb, klcl)`` — precipitation ``(...)``
    [kg/m^2], the T/q increments ``(..., K)``, CAPE/CIN ``(...)``, and 0-based
    ``klzb``/``klcl`` indices ``(...)``.
    """
    flat = tin.shape[:-1]
    K = tin.shape[-1]
    tin2 = tin.reshape(-1, K)
    qin2 = qin.reshape(-1, K)
    pf2 = p_full.reshape(-1, K)
    ph2 = p_half.reshape(-1, K + 1)

    def one(t, q, pf, ph):
        tdel, qdel, rain, cape, cin, klzb, klcl, _tr, _qr = _betts_miller_column(
            bp, dt, t, q, pf, ph)
        return tdel, qdel, rain, cape, cin, klzb, klcl

    tdel, qdel, rain, cape, cin, klzb, klcl = jax.vmap(
        one, in_axes=(0, 0, 0, 0))(tin2, qin2, pf2, ph2)
    return (rain.reshape(flat), tdel.reshape(flat + (K,)), qdel.reshape(flat + (K,)),
            cape.reshape(flat), cin.reshape(flat), klzb.reshape(flat), klcl.reshape(flat))
