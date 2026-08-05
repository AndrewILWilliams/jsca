"""Tier-1 tests for compute_vert_coord against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_vert_coordinate_reference.F90``,
which compiles Isca's actual ``vert_coordinate.F90`` unmodified (fms_mod stubbed;
the namelist-read 'input' path is never exercised). Regeneration recipe in that
file's header. The a/b coefficients are 1-D interface arrays (length K+1), so no
layout reconciliation is needed.
"""

from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401
from jsca.dycore import compute_vert_coord

FIXTURE = Path(__file__).parent / "fixtures" / "vert_coordinate_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="vert_coordinate fixtures not generated"
)

# tag -> (option, num_levels)
CASES = {
    "even": ("even_sigma", 25),
    "uneven": ("uneven_sigma", 25),
    "hybrid": ("hybrid", 25),
    "mcm": ("mcm", 14),
    "v197": ("v197", 18),
}


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.mark.parametrize("tag", list(CASES))
def test_vert_coordinate(fx, tag):
    option, num_levels = CASES[tag]
    scale_heights, surf_res, exponent, p_press, p_sigma, reference_press = fx["vc_meta"]
    a, b = compute_vert_coord(
        option, num_levels,
        scale_heights=scale_heights, surf_res=surf_res, exponent=exponent,
        p_press=p_press, p_sigma=p_sigma, reference_press=reference_press,
    )
    # exp-bearing (uneven/hybrid) -> rtol 1e-13; the rest are exact (0 == 0)
    np.testing.assert_allclose(a, fx[f"vc_{tag}_a"], rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(b, fx[f"vc_{tag}_b"], rtol=1e-13, atol=0.0)


def test_even_sigma_bounds(fx):
    """a and b combine as p_half = a*p_ref + b*p_surf (F90 L63); sanity on shapes/bounds."""
    a, b = compute_vert_coord("even_sigma", 25)
    assert a.shape == b.shape == (26,)
    assert b[0] == 0.0 and b[-1] == 1.0  # sigma runs 0 (top) -> 1 (surface)
    assert np.all(a == 0.0)  # pure sigma
