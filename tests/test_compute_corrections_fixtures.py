"""Tier-1 tests for compute_corrections against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_compute_corrections_reference.F90``.
``compute_corrections`` is a *private* routine of the (un-stubbable)
spectral_dynamics module, so its body is compiled VERBATIM from the pinned source
into a scaffold module that supplies its module-variable environment and links the
REAL global_integral.F90 / press_and_geopot.F90 (as the global_integral fixture
does). Regeneration recipe in the wrapper file's header.

Fortran storage is ``(lon, lat, level)``; the port uses jsca's
``(..., nlat, nlon[, K])`` grid convention (the global integrals weight by
Gaussian latitude), so fixtures transpose lon<->lat. A minimal ``TransformParams``
carrying the dumped Gaussian weights reproduces the Fortran area mean exactly.

Only the dry Held-Suarez path (mass + energy corrections) is exercised; the
wet-model water correction is a documented follow-up.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import jsca  # noqa: F401
from jsca.dycore import compute_corrections
from jsca.grid.transforms import TransformParams

FIXTURE = Path(__file__).parent / "fixtures" / "compute_corrections_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="compute_corrections fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def _params(fx):
    """Minimal TransformParams with the fixture's Gaussian weights (only wts_lat /
    nlon are read by the global integrals)."""
    nlon = int(fx["cc_meta"][0])
    return TransformParams(
        legendre=None, legendre_wts=None, sin_lat=None,
        wts_lat=jnp.asarray(fx["cc_wts_lat"]),
        mask_prognostic=None, mask_storage=None, lap_eig=None, coeffs=None,
        nlon=nlon, num_fourier=0,
    )


def _t3(a):
    """Fortran (lon, lat, lev) -> jsca (lat, lon, lev)."""
    return np.transpose(a, (1, 0, 2))


def test_compute_corrections(fx):
    p = _params(fx)
    pk, bk = fx["cc_pk"], fx["cc_bk"]
    psg, tg, lnps00, ts00 = compute_corrections(
        p, pk, bk,
        fx["cc_psg_in"].T, _t3(fx["cc_ug"]), _t3(fx["cc_vg"]), _t3(fx["cc_tg_in"]),
        float(fx["cc_lnps00_in"][0]), jnp.asarray(fx["cc_ts00_in"]),
        float(fx["cc_mean_sp_prev"][0]), float(fx["cc_mean_en_prev"][0]),
    )
    np.testing.assert_allclose(np.asarray(psg), fx["cc_psg_out"].T, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(tg), _t3(fx["cc_tg_out"]), rtol=1e-12, atol=1e-9)
    np.testing.assert_allclose(float(lnps00), float(fx["cc_lnps00_out"][0]), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(ts00), fx["cc_ts00_out"], rtol=1e-12, atol=1e-9)


def test_mass_is_corrected(fx):
    """After the mass correction, the area-mean surface pressure equals the target."""
    p = _params(fx)
    from jsca.grid.transforms import area_weighted_global_mean
    psg, *_ = compute_corrections(
        p, fx["cc_pk"], fx["cc_bk"],
        fx["cc_psg_in"].T, _t3(fx["cc_ug"]), _t3(fx["cc_vg"]), _t3(fx["cc_tg_in"]),
        float(fx["cc_lnps00_in"][0]), jnp.asarray(fx["cc_ts00_in"]),
        float(fx["cc_mean_sp_prev"][0]), float(fx["cc_mean_en_prev"][0]),
        do_energy_correction=False,
    )
    mean_after = float(area_weighted_global_mean(p, jnp.asarray(psg)))
    np.testing.assert_allclose(mean_after, float(fx["cc_mean_sp_prev"][0]), rtol=1e-12)


def test_water_correction_not_supported(fx):
    p = _params(fx)
    with pytest.raises(NotImplementedError):
        compute_corrections(
            p, fx["cc_pk"], fx["cc_bk"],
            fx["cc_psg_in"].T, _t3(fx["cc_ug"]), _t3(fx["cc_vg"]), _t3(fx["cc_tg_in"]),
            float(fx["cc_lnps00_in"][0]), jnp.asarray(fx["cc_ts00_in"]),
            float(fx["cc_mean_sp_prev"][0]), float(fx["cc_mean_en_prev"][0]),
            do_water_correction=True,
        )
