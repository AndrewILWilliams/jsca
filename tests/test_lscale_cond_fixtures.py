"""Tier-1 tests for large-scale condensation against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_lscale_cond_reference.F90``,
which compiles the **unmodified** ``lscale_cond.F90`` and the real
``sat_vapor_pres.F90`` wrapper with the Frierson namelist
(``do_simple=.true., do_evap=.true., hc=1.0``) injected through FMS's
internal-file buffer, and dumps ``(tdel, qdel, rain)`` for a column set built to
exercise condensation, partial re-evaporation, and surviving surface rain.

The Fortran arrays are ``(i, j, k)`` with the level axis last — exactly jsca's
column-physics layout — so they feed :func:`lscale_cond` directly (the ``(i, j)``
axes are batched).

Tolerance: the only inexactness is the documented ``sat_vapor_pres``
table-vs-closed-form deviation (the fixture's ``compute_qs`` interpolates a
lookup table; jsca evaluates the closed form). That ~2e-7 relative difference in
``qsat`` enters the adjustment through the **near-cancellation** ``qsat - qin``
(qin = RH·qsat with RH ~ 1.1 in the condensing layers), which amplifies it by
~1/(RH-1) ~ 10x, so the observed residual on ``tdel``/``qdel``/``rain`` is ~3e-6
relative. Held at 1e-5. ``tdel`` and ``qdel`` carry the *same* relative error
(``tdel = -hlcp·qdel``); the condensation and re-evaporation arithmetic itself is
exact to machine precision. This residual would vanish once ``compute_qs`` shares
jsca's closed form; it is a property of the fixture's table, not the port.
"""
from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.physics import lscale_cond

FIXTURE = Path(__file__).parent / "fixtures" / "lscale_cond_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="lscale_cond fixtures not generated"
)

RTOL = 1e-5   # sat_vapor_pres table deviation, amplified by (qsat-qin) cancellation
ATOL = 1e-9   # absolute floor for near-zero increments


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _run(fx):
    rain, tdel, qdel = lscale_cond(
        fx["lc_tin"], fx["lc_qin"], fx["lc_pfull"], fx["lc_phalf"],
        hc=1.0, do_evap=True,
    )
    return np.asarray(rain), np.asarray(tdel), np.asarray(qdel)


def test_qdel_matches_fortran(fx):
    _, _, qdel = _run(fx)
    assert np.allclose(qdel, fx["lc_qdel"], rtol=RTOL, atol=ATOL)


def test_tdel_matches_fortran(fx):
    _, tdel, _ = _run(fx)
    assert np.allclose(tdel, fx["lc_tdel"], rtol=RTOL, atol=ATOL)


def test_rain_matches_fortran(fx):
    rain, _, _ = _run(fx)
    # The fixture is designed to have non-trivial surviving rain in some columns.
    assert fx["lc_rain"].max() > 1e-3
    assert np.allclose(rain, fx["lc_rain"], rtol=RTOL, atol=ATOL)


def test_exercises_condensation_and_reevaporation(fx):
    """Guard that the fixture actually spans both branches (not a degenerate case)."""
    assert (fx["lc_qdel"] < 0).sum() > 0   # condensation (drying) cells
    assert (fx["lc_qdel"] > 0).sum() > 0   # re-evaporation (moistening) cells
