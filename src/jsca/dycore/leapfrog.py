"""Leapfrog time stepping with Robert–Asselin–Williams (RAW) filtering.

Faithful port of Isca's ``src/atmos_spectral/model/leapfrog.F90``. Notation:
``a`` holds the time levels in its **last** axis (as in the Fortran storage
``a(..., ntime)``); ``previous/current/future`` are 0-based static indices;
``robert_coeff`` is the filter strength (nu) and ``raw_filter_coeff`` is
Williams' alpha (alpha = 1 recovers the classic Robert–Asselin filter; see
Williams 2011, doi:10.1175/2010MWR3601.1).

Scheme (combined form, ``leapfrog``):

    P            = a^{t-dt} - 2 a^t                       (pre-update)
    a^{t+dt}     = a^{t-dt} + 2 dt * (da/dt)
    a^t   <- a^t + nu * alpha * (P + a^{t+dt})
    a^{t+dt} <- a^{t+dt} + nu * (alpha - 1) * (P + a^{t+dt})

The two-level split (``leapfrog_2level_a`` then ``..._b``) is used by
``spectral_dynamics`` when the future value is corrected (semi-implicit
adjustment) between the two calls; ``P`` from step A must be threaded to B.

Fortran quirk preserved deliberately: the *real* (grid-tracer) variant of
``leapfrog_2level_A`` omits the ``raw_filter_coeff`` factor on the
current-level update (leapfrog.F90 line 128), while the complex variant
applies it (line 77). ``leapfrog_2level_a_real`` reproduces that. Do not
"fix" this without a matching change in the validated Fortran baseline.

All functions are pure (return the updated array) and jit/vmap-safe as long
as the index arguments are static Python ints.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

Array = jnp.ndarray


def apply_q_decrease_only(q: Array) -> Array:
    """Clamp a humidity column so it never increases with height — port of the
    ``q_decrease_only`` block of the single-column ``leapfrog_3d_real``
    (``atmos_column/column.F90`` L782-792).

    Isca's SCM offers ``q_decrease_only`` to stop the stratospheric humidity from
    growing upward (a common artefact when the dynamical moisture sink is absent).
    The Fortran sweeps levels from the second-lowest **up** to the top and, wherever
    a level's humidity exceeds the level just below it, copies the below value up::

        do k = K-1, 1, -1
          if q(k) > q(k+1) then q(k) = q(k+1)

    With the level axis last (``k = 0`` top … ``K-1`` surface) and applied per
    column, the sequential upward sweep is exactly a reverse cumulative minimum
    from the surface: ``q[..., k] = min(q[..., k], q[..., k+1], …, q[..., K-1])``.

    **Deliberate, documented deviation (rule 1).** The Fortran takes its
    *decision* from the first column ``q(1,1,k)`` yet assigns to every column
    ``q(:,:,k) = q(:,:,k+1)`` — a quirk that is only self-consistent for a genuine
    single column (``lat_max = lon_max = 1``), the case the SCM is built for. Here
    the clamp is applied **independently per column** (each column uses its own
    profile). For the single-column case the two are identical; for a multi-column
    run this is the physically sensible interpretation rather than Isca's
    first-column-decides-for-all behaviour.
    """
    # cummin needs a non-negative axis; the level axis is the last one.
    return jax.lax.cummin(q, axis=q.ndim - 1, reverse=True)


def leapfrog(
    a: Array,
    dt_a: Array,
    previous: int,
    current: int,
    future: int,
    delta_t: float,
    robert_coeff: float,
    raw_filter_coeff: float,
) -> Array:
    """Combined leapfrog + RAW filter (port of ``leapfrog_3d_complex``)."""
    p = a[..., previous] - 2.0 * a[..., current]
    if previous == current:  # first (Euler-ish) step of a start-up sequence
        fut = a[..., previous] + delta_t * dt_a
        cur = a[..., current] + robert_coeff * (p + fut) * raw_filter_coeff
    else:
        cur = a[..., current] + robert_coeff * p * raw_filter_coeff
        fut = a[..., previous] + delta_t * dt_a
        cur = cur + robert_coeff * fut * raw_filter_coeff
    fut = fut + robert_coeff * (p + fut) * (raw_filter_coeff - 1.0)
    return a.at[..., current].set(cur).at[..., future].set(fut)


def leapfrog_2level_a(
    a: Array,
    dt_a: Array,
    previous: int,
    current: int,
    future: int,
    delta_t: float,
    robert_coeff: float,
    raw_filter_coeff: float,
) -> tuple[Array, Array]:
    """First half-step (port of ``leapfrog_2level_A_3d_complex``).

    Returns ``(a_updated, part_filt)`` where ``part_filt = a_prev - 2 a_curr``
    must be passed unchanged to :func:`leapfrog_2level_b`.
    """
    p = a[..., previous] - 2.0 * a[..., current]
    cur = a[..., current] + robert_coeff * p * raw_filter_coeff
    fut = a[..., previous] + delta_t * dt_a
    return a.at[..., current].set(cur).at[..., future].set(fut), p


def leapfrog_2level_a_real(
    a: Array,
    dt_a: Array,
    previous: int,
    current: int,
    future: int,
    delta_t: float,
    robert_coeff: float,
    raw_filter_coeff: float,  # accepted but unused on the current-level update, as in Fortran
) -> tuple[Array, Array]:
    """Port of ``leapfrog_2level_A_3d_real`` — see module docstring quirk note."""
    del raw_filter_coeff
    p = a[..., previous] - 2.0 * a[..., current]
    cur = a[..., current] + robert_coeff * p
    fut = a[..., previous] + delta_t * dt_a
    return a.at[..., current].set(cur).at[..., future].set(fut), p


def leapfrog_2level_b(
    a: Array,
    part_filt_a: Array,
    current: int,
    future: int,
    robert_coeff: float,
    raw_filter_coeff: float,
) -> Array:
    """Second half-step (port of ``leapfrog_2level_B_3d_{complex,real}``)."""
    fut0 = a[..., future]
    cur = a[..., current] + robert_coeff * fut0 * raw_filter_coeff
    fut = fut0 + robert_coeff * (part_filt_a + fut0) * (raw_filter_coeff - 1.0)
    return a.at[..., current].set(cur).at[..., future].set(fut)
