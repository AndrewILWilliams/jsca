"""Tier-1 tests for the bucket-hydrology reservoir stepping against real Fortran.

The Manabe soil-moisture bucket is marched by ``idealized_moist_phys.F90``
L1401-1428: a grid-space leapfrog + RAW filter (coefficients
``robert_bucket=0.04``, ``raw_bucket=0.53``) followed by the non-negativity clamp
and the over-land runoff clamp. The fixture
(``fortran_instrumentation/dump_bucket_stepping_reference.F90``) compiles that
block VERBATIM (extracted as ``bucket_stepping_body.inc``) and fires it twice:

* ``step``  — the normal leapfrog step (``previous /= current``);
* ``start`` — the first Euler-ish step (``previous == current``).

Inputs span empty buckets (exercise the ``<= 0`` clamp), below-capacity depths
(plain leapfrog) and over-capacity depths over both land (runoff fires) and ocean
(runoff must NOT fire). jsca matches Isca to machine precision because the
stepping *is* the leapfrog+RAW filter already ported in
:mod:`jsca.dycore.leapfrog` plus two elementwise clamps.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics.bucket import bucket_step

FIXTURE = Path(__file__).parent / "fixtures" / "bucket_stepping_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="bucket stepping fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _run(fx, prev_in, curr_in, previous, current, future):
    """Assemble the (..., 2) bucket array and run one bucket_step."""
    n = prev_in.shape[0]
    bd = np.zeros((n, 2))
    bd[:, previous] = prev_in
    bd[:, current] = curr_in
    bd = jnp.asarray(bd)
    land = fx["bk_land"] > 0.5
    out = bucket_step(
        bd, fx["bk_cond"], fx["bk_conv"], fx["bk_lh"], land,
        previous, current, future, float(fx["bk_max"][0]),
        robert_coeff=float(fx["bk_robert"][0]),
        raw_filter_coeff=float(fx["bk_raw"][0]),
    )
    return np.asarray(out)


def test_normal_step_matches_fortran(fx):
    """previous /= current: future=previous, current is RAW-filtered."""
    out = _run(fx, fx["bk_step_prev_in"], fx["bk_step_curr_in"],
               previous=0, current=1, future=0)
    assert np.allclose(out[:, 0], fx["bk_step_future"], rtol=1e-13, atol=1e-15)
    assert np.allclose(out[:, 1], fx["bk_step_current"], rtol=1e-13, atol=1e-15)


def test_start_step_matches_fortran(fx):
    """previous == current: first Euler-ish step, future=other slot."""
    # Fortran fires with previous=current=slot0, future=slot1.
    out = _run(fx, fx["bk_start_in"], fx["bk_start_in"],
               previous=0, current=0, future=1)
    assert np.allclose(out[:, 1], fx["bk_start_future"], rtol=1e-13, atol=1e-15)
    assert np.allclose(out[:, 0], fx["bk_start_current"], rtol=1e-13, atol=1e-15)


def test_runoff_only_over_land(fx):
    """Over-capacity depths clamp to the cap on land but are left alone on ocean."""
    land = fx["bk_land"] > 0.5
    cap = float(fx["bk_max"][0])
    fut = fx["bk_step_future"]
    # No land point exceeds the cap...
    assert np.all(fut[land] <= cap + 1e-12)
    # ...but at least one ocean point legitimately does (runoff skipped there).
    assert np.any(fut[~land] > cap)


def test_reservoir_never_negative(fx):
    """The <=0 clamp guarantees a non-negative reservoir at every level."""
    assert np.all(fx["bk_step_future"] >= 0.0)
    assert np.all(fx["bk_step_current"] >= 0.0)
    # And the fixture actually drives some points to exactly zero.
    assert np.any(fx["bk_step_future"] == 0.0)
