"""Tier-1 tests for the water (humidity) conservation correction vs real Fortran.

Fixtures come from ``fortran_instrumentation/dump_water_correction_reference.F90``,
which compiles the **verbatim** ``compute_corrections`` body from the pinned
source (via ``compute_corrections_wrapper.F90``) and runs *only* the water branch
(``do_mass_correction=.false., do_energy_correction=.false.,
do_water_correction=.true., dry_model=.false.``) with a **grid** humidity tracer
and ``water_correction_limit = 200 hPa`` — the Frierson settings, which exercise
the MiMA pressure-limit remapping (some levels above the limit are left
uncorrected). Dumps the humidity before/after the rescaling.

Pure arithmetic (mass-weighted global integrals + a scalar rescale), so it matches
to machine precision.

Fortran storage is ``(lon, lat, lev)``; the port uses jsca's ``(nlat, nlon, K)``
grid convention (the global integral weights by Gaussian latitude), so fixtures
transpose lon<->lat. A minimal ``TransformParams`` carries the dumped Gaussian
weights.
"""
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401
from jsca.dycore import water_correction
from jsca.grid.transforms import TransformParams

FIXTURE = Path(__file__).parent / "fixtures" / "water_correction_reference.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="water_correction fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _params(fx):
    nlon = int(fx["wc_meta"][0])
    return TransformParams(
        legendre=None, legendre_wts=None, sin_lat=None,
        wts_lat=jnp.asarray(fx["wc_wts_lat"]),
        mask_prognostic=None, mask_storage=None, lap_eig=None, coeffs=None,
        nlon=nlon, num_fourier=0,
    )


def _t3(a):
    """Fortran (lon, lat, lev) -> jsca (lat, lon, lev)."""
    return np.transpose(a, (1, 0, 2))


@pytest.fixture(scope="module")
def result(fx):
    p = _params(fx)
    q_out, factor = water_correction(
        p, fx["wc_pk"], fx["wc_bk"],
        _t3(fx["wc_q_in"]), fx["wc_psg"].T, _t3(fx["wc_p_full"]),
        float(fx["wc_mean_water_prev"][0]), float(fx["wc_limit"][0]),
    )
    return {"q": np.asarray(q_out), "factor": np.asarray(factor)}


def test_matches_fortran(result, fx):
    np.testing.assert_allclose(result["q"], _t3(fx["wc_q_out"]), rtol=1e-12, atol=1e-18)


def test_water_is_conserved(result, fx):
    """After the correction, the global-mean humidity equals the target."""
    from jsca.dycore.global_integral import mass_weighted_global_integral
    p = _params(fx)
    mean_after = float(mass_weighted_global_integral(
        p, fx["wc_pk"], fx["wc_bk"], jnp.asarray(result["q"]), jnp.asarray(fx["wc_psg"].T)))
    np.testing.assert_allclose(mean_after, float(fx["wc_mean_water_prev"][0]), rtol=1e-11)


def test_limit_leaves_high_levels_untouched(result, fx):
    """High, thin levels (p_full < limit) are left exactly unchanged; only the
    p_full >= limit region is rescaled."""
    q_in = _t3(fx["wc_q_in"])
    p_full = _t3(fx["wc_p_full"])
    above = p_full < float(fx["wc_limit"][0])
    np.testing.assert_array_equal(result["q"][above], q_in[above])
    # and the corrected region actually changed
    below = p_full >= float(fx["wc_limit"][0])
    assert np.any(result["q"][below] != q_in[below])
