"""Astronomy — solar zenith angle and insolation geometry.

Faithful port of the diurnal-cycle path of Isca's
``src/shared/astronomy/astronomy.f90`` — the engine behind
``two_stream_gray_rad``'s ``do_seasonal`` insolation. Given the latitude,
longitude, time of day (``gmt``) and orbital position (``time_since_ae``), it
returns the cosine of the solar zenith angle (instantaneous, or time-averaged
over a radiation step ``dt``), the daylight fraction, and the Earth–Sun distance
factor ``rrsun = (a/r)^2``.

Only the ``diurnal_solar`` path is ported (the one the seasonal grey-radiation
uses). The annual-mean / calendar (``*_cal``) and ``set_orbital_parameters``
paths are not.

Pipeline (F90 ``diurnal_solar_2d`` L1123-1410, orbital helpers L3138-3519):

* ``orbit`` (:func:`build_orbit_angle`) precomputes a table of orbital angle vs
  orbital time by RK4-integrating ``d(angle)/dt ∝ r_inv_squared`` (Kepler's 2nd
  law). For the default zero eccentricity the table is exactly linear.
* ``angle(t)`` interpolates that table; ``declination(ang)`` and
  ``r_inv_squared(ang)`` give the solar declination and distance factor.
* ``half_day(lat, dec)`` is the half-length of the daylit arc.
* ``diurnal_solar`` assembles ``cosz = max(0, aa + bb·cos(local_time))`` with
  ``aa = sin(lat)sin(dec)``, ``bb = cos(lat)cos(dec)``; when ``dt`` is given it
  returns the analytic time-average over ``[t, t+dt]`` via the nine day/night
  overlap cases (F90 L1274-1364), with the denominators fixed to ``(tt - t)``
  (the corrected true-average behaviour, F90 st_doc L1258-1272).

Convention: latitudes/longitudes in radians; ``gmt``/``time_since_ae``/``dt`` in
radians (``2π`` = one day / one year respectively). ``lat``/``lon`` are batched
arrays of the same shape; ``gmt``/``time_since_ae``/``dt`` are scalars, so
``ang``/``dec``/``rrsun`` are scalars (as in the Fortran, which evaluates them
once per call). No documented deviations: pure trig, so fixtures hit the
log/exp tolerance band.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

Array = jnp.ndarray

DEG_TO_RAD = np.pi / 180.0
TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class AstronomyParams:
    """Static orbital configuration (hashable; Isca ``astronomy_nml`` defaults).

    ``ecc`` = 0 is Isca's simplified-model default (a circular orbit, so
    ``rrsun`` ≡ 1 and the orbital-angle table is linear); ``obliq``/``per`` are
    the present-day Earth values. ``num_angles`` sets the resolution of the
    precomputed orbital table.
    """

    ecc: float = 0.0          # eccentricity (Isca default 0; F90 L145)
    obliq: float = 23.439     # obliquity [deg] (F90 L147)
    per: float = 102.932      # longitude of perihelion [deg] (F90 L148)
    num_angles: int = 3600    # orbital-table resolution (F90 L160)


def _r_inv_squared_np(ang: np.ndarray, ecc: float, per: float) -> np.ndarray:
    """``(a/r)^2`` at orbital angle ``ang`` (F90 ``r_inv_squared`` L3207-3257)."""
    rad_per = per * DEG_TO_RAD
    r = (1.0 - ecc ** 2) / (1.0 + ecc * np.cos(ang - rad_per))
    return r ** (-2)


def build_orbit_angle(params: AstronomyParams) -> Array:
    """Precompute the orbital-angle table (F90 ``orbit`` L3138-3181).

    Integrates ``d(angle)/d(time) = norm · r_inv_squared(angle)`` from the
    autumnal equinox with RK4, ``norm = sqrt(1 - ecc^2)``, over ``num_angles``
    equal time steps. Returns ``(num_angles + 1,)``. Runs once at init (NumPy —
    not on the step path); the result is a static numeric table for
    :func:`angle`.
    """
    n = params.num_angles
    orb = np.zeros(n + 1)
    dt = (TWOPI / n) * np.sqrt(1.0 - params.ecc ** 2)
    for k in range(1, n + 1):
        a0 = orb[k - 1]
        d1 = dt * _r_inv_squared_np(a0, params.ecc, params.per)
        d2 = dt * _r_inv_squared_np(a0 + 0.5 * d1, params.ecc, params.per)
        d3 = dt * _r_inv_squared_np(a0 + 0.5 * d2, params.ecc, params.per)
        d4 = dt * _r_inv_squared_np(a0 + d3, params.ecc, params.per)
        orb[k] = a0 + d1 / 6.0 + d2 / 3.0 + d3 / 3.0 + d4 / 6.0
    return jnp.asarray(orb)


def angle(t: Array, orb_angle: Array, num_angles: int) -> Array:
    """Orbital position at orbital time ``t`` (F90 ``angle`` L3283-3344).

    Linearly interpolates the precomputed table, wrapping to ``[0, 2π)``.
    """
    norm_time = t * num_angles / TWOPI
    i = jnp.floor(norm_time).astype(jnp.int32)
    i = jnp.mod(i, num_angles)
    x = norm_time - jnp.floor(norm_time)
    ang = (1.0 - x) * orb_angle[i] + x * orb_angle[i + 1]
    return jnp.mod(ang, TWOPI)


def declination(ang: Array, obliq: float) -> Array:
    """Solar declination at orbital angle ``ang`` (F90 ``declination`` L3364)."""
    rad_obliq = obliq * DEG_TO_RAD
    return jnp.arcsin(-jnp.sin(rad_obliq) * jnp.sin(ang))


def r_inv_squared(ang: Array, ecc: float, per: float) -> Array:
    """``(a/r)^2`` at orbital angle ``ang`` (F90 ``r_inv_squared`` L3207)."""
    rad_per = per * DEG_TO_RAD
    r = (1.0 - ecc ** 2) / (1.0 + ecc * jnp.cos(ang - rad_per))
    return r ** (-2)


def half_day(lat: Array, dec: Array) -> Array:
    """Half the daylit arc [radians] (F90 ``half_day_2d`` L3461-3519).

    ``h = acos(-tan(lat)·tan(dec))``, saturating to ``π`` (polar day) or ``0``
    (polar night). Latitude is nudged off the poles so ``tan`` is finite.
    """
    eps = 1.0e-5
    lat_adj = jnp.where(lat == 0.5 * np.pi, lat - eps, lat)
    lat_adj = jnp.where(lat_adj == -0.5 * np.pi, lat_adj + eps, lat_adj)
    cos_half = -jnp.tan(lat_adj) * jnp.tan(dec)
    h = jnp.arccos(jnp.clip(cos_half, -1.0, 1.0))
    h = jnp.where(cos_half <= -1.0, np.pi, h)     # polar day
    h = jnp.where(cos_half >= 1.0, 0.0, h)        # polar night
    return h


def diurnal_solar(params: AstronomyParams, orb_angle: Array, lat: Array,
                  lon: Array, gmt: Array, time_since_ae: Array,
                  dt: Array | None = None):
    """Cosine of the solar zenith angle etc. (F90 ``diurnal_solar_2d``).

    Returns ``(cosz, fracday, rrsun)``. ``cosz`` and ``fracday`` have ``lat``'s
    shape; ``rrsun`` is a scalar. With ``dt`` given, ``cosz`` is the true time
    average over the interval ``[t, t+dt]`` (``allow_negative_cosz`` is always
    ``False`` here, matching the grey-radiation caller).
    """
    ang = angle(time_since_ae, orb_angle, params.num_angles)
    dec = declination(ang, params.obliq)
    rrsun = r_inv_squared(ang, params.ecc, params.per)

    aa = jnp.sin(lat) * jnp.sin(dec)
    bb = jnp.cos(lat) * jnp.cos(dec)

    # local time, forced into [-pi, pi)  (F90 L1201-1203)
    t = gmt + lon - np.pi
    t = jnp.where(t >= np.pi, t - TWOPI, t)
    t = jnp.where(t < -np.pi, t + TWOPI, t)

    h = half_day(lat, dec)

    if dt is None:
        # instantaneous (F90 L1382-1398, Lallow_negative=False)
        day = jnp.abs(t) < h
        cosz = jnp.where(day, aa + bb * jnp.cos(t), 0.0)
        fracday = jnp.where(day, 1.0, 0.0)
    else:
        # time average over [t, tt=t+dt]: the nine day/night overlap cases
        # (F90 L1274-1364), denominators fixed to (tt - t) for a true average.
        tt = t + dt
        st, stt, sh = jnp.sin(t), jnp.sin(tt), jnp.sin(h)
        denom = tt - t                                  # = dt
        cosz = jnp.zeros_like(lat)

        # case 1: entire period before sunrise -> 0 (cosz already 0)
        # case 2: begins before sunrise, ends in daylight
        m = (t < -h) & (jnp.abs(tt) <= h)
        cosz = jnp.where(m, aa * (tt + h) / denom + bb * (stt + sh) / denom, cosz)
        # case 3: begins before sunrise, ends after sunset (full day)
        m = (t < -h) & (h != 0.0) & (h < tt)
        cosz = jnp.where(m, aa * (2.0 * h) / denom + bb * (sh + sh) / denom, cosz)
        # case 4: entirely within daylight
        m = (jnp.abs(t) <= h) & (jnp.abs(tt) <= h)
        cosz = jnp.where(m, aa + bb * (stt - st) / denom, cosz)
        # case 5: begins in daylight, ends after sunset
        m = (jnp.abs(t) <= h) & (h < tt)
        cosz = jnp.where(m, aa * (h - t) / denom + bb * (sh - st) / denom, cosz)
        # case 6: extends past the next day's sunrise
        m = (TWOPI - h < tt) & (t <= h)
        cosz = jnp.where(
            m, aa * ((tt + 2.0 * h - t - TWOPI) / denom)
            + bb * ((sh - st) / denom + (stt + sh) / denom), cosz)
        # case 7: after sunset, before next sunrise -> 0
        m = (h < t) & (TWOPI - h >= tt)
        cosz = jnp.where(m, 0.0, cosz)
        # case 8: after sunset, ends after next sunrise but before next sunset
        m = (h < t) & (TWOPI - h < tt) & (tt < TWOPI + h)
        cosz = jnp.where(m, aa * (tt + h - TWOPI) / denom + bb * (stt + sh) / denom, cosz)
        # case 9: after sunset, ends after the next day's sunset
        m = (h < t) & (TWOPI - h < tt) & (tt > TWOPI + h)
        cosz = jnp.where(m, aa * (2.0 * h) / denom + bb * (sh + sh) / denom, cosz)

        # day fraction (F90 L1370-1378); the twopi-h line is additive
        fracday = jnp.zeros_like(lat)
        fracday = jnp.where((t < -h) & (tt < -h), 0.0, fracday)
        fracday = jnp.where((t < -h) & (jnp.abs(tt) <= h), (tt + h) / dt, fracday)
        fracday = jnp.where((t < -h) & (h < tt), (h + h) / dt, fracday)
        fracday = jnp.where((jnp.abs(t) <= h) & (jnp.abs(tt) <= h), (tt - t) / dt, fracday)
        fracday = jnp.where((jnp.abs(t) <= h) & (h < tt), (h - t) / dt, fracday)
        fracday = jnp.where(h < t, 0.0, fracday)
        fracday = jnp.where(TWOPI - h < tt, fracday + (tt + h - TWOPI) / dt, fracday)

    cosz = jnp.maximum(0.0, cosz)
    return cosz, fracday, rrsun
