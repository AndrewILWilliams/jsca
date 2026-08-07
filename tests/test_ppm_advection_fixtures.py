"""Tier-1 tests for the PPM (finite-volume parabolic) vertical advection scheme.

Fixtures come from ``fortran_instrumentation/dump_ppm_advection_reference.F90``,
which compiles the **unmodified** ``vert_advection.F90`` and runs
``FINITE_VOLUME_PARABOLIC`` in both equation forms over a column set spanning two
Courant regimes:

* gentle winds of both signs (``|Courant| < 1``) — the single-cell PPM flux;
* strong, positive-only winds (``Courant`` up to ~9) — the multi-cell Courant>1
  extension.

The strong winds are positive-only on purpose: Isca's PPM has an out-of-bounds
read in the *negative*-wind Courant>1 branch (its walk exits on ``kk==ks`` while
incrementing ``kk`` toward ``ke``), so its output there is undefined. The jsca
port clamps the departure cell at ``ke`` (the intended behaviour), matching Isca
on every well-defined path; that clamped ``w<0`` & ``Courant>1`` path is not
fixture-checkable and is not reached by Frierson (sub-unity vertical Courant).

Pure arithmetic (parabolic reconstruction + Colella-Woodward limiter + a
departure-point flux integral), so it matches to machine precision.
"""
from pathlib import Path

import jax
import numpy as np
import pytest

import jsca  # noqa: F401
from jsca.dycore.vert_advection import (
    ADVECTIVE_FORM,
    FINITE_VOLUME_PARABOLIC,
    FLUX_FORM,
    vert_advection,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ppm_advection_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="ppm_advection fixtures not generated"
)

DT = 600.0


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.mark.parametrize("form,key", [(ADVECTIVE_FORM, "ppm_adv"), (FLUX_FORM, "ppm_flux")])
def test_matches_fortran(fx, form, key):
    out = vert_advection(DT, fx["ppm_w"], fx["ppm_dz"], fx["ppm_r"],
                         scheme=FINITE_VOLUME_PARABOLIC, form=form)
    np.testing.assert_allclose(np.asarray(out), fx[key], rtol=1e-12, atol=1e-18)


def test_courant_gt1_branch_is_exercised(fx):
    """The strong-wind columns must actually reach Courant > 1 (else the multi-cell
    extension is untested)."""
    w, dz = fx["ppm_w"], fx["ppm_dz"]
    k = fx["ppm_r"].shape[-1]
    cn = np.maximum(
        DT * np.maximum(w[..., 1:k], 0.0) / dz[..., : k - 1],
        -DT * np.minimum(w[..., 1:k], 0.0) / dz[..., 1:k],
    )
    assert cn.max() > 1.0


def test_jit(fx):
    f = jax.jit(lambda w, dz, r: vert_advection(
        DT, w, dz, r, scheme=FINITE_VOLUME_PARABOLIC, form=ADVECTIVE_FORM))
    out = np.asarray(f(fx["ppm_w"], fx["ppm_dz"], fx["ppm_r"]))
    np.testing.assert_allclose(out, fx["ppm_adv"], rtol=1e-12, atol=1e-18)
