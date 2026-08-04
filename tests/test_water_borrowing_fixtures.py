"""Tier-1 tests for water_borrowing against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_water_borrowing_reference.F90``,
which compiles Isca's actual ``water_borrowing.F90`` unmodified (transforms_mod
stubbed for ``get_grid_domain``). Regeneration recipe in that file's header.

Fortran grid storage is ``(lon, lat, level)``; the port keeps ``(lat, lon, level)``,
so fixtures transpose lon<->lat. The Fortran sweeps longitude sequentially with a
direction set by ``current``'s parity, but reads only the original ``qg`` — so the
result is direction-independent (the ``even`` and ``odd`` fixtures differ only at
the last bit). The vectorised port matches both to near machine precision.
"""

from pathlib import Path

import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.dycore import water_borrowing

FIXTURE = Path(__file__).parent / "fixtures" / "water_borrowing_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="water_borrowing fixtures not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def latlon(a):
    """Fortran (lon, lat, level) -> port (lat, lon, level)."""
    return np.moveaxis(a, 0, 1)


@pytest.mark.parametrize("tag", ["even", "odd"])
def test_water_borrowing(fx, tag):
    dt = float(fx["wb_meta"][3])
    out = water_borrowing(
        latlon(fx["wb_dt_qg_in"]), latlon(fx["wb_qg"]), latlon(fx["wb_p_half"]), dt
    )
    ref = latlon(fx[f"wb_dt_qg_{tag}"])
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-14, atol=1e-18)


def test_holes_present(fx):
    """The fixture actually exercises the borrowing (negative-humidity cells exist)."""
    assert int((fx["wb_qg"] < 0.0).sum()) > 0
