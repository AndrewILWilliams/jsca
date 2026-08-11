"""Test the idealized-continents land mask against Isca's ``land_generator_fn``.

The golden fixture ``continents_land_t42.npz`` is produced by executing the
*verbatim* ``land_mode='continents'`` block of Isca's
``src/extra/python/isca/land_generator_fn.py`` (L84-105) on Isca's own T42
Gaussian grid (``gfdl_grid_files/t42.nc``). :func:`continents_land_mask` must
reproduce it bit-for-bit — the mask is pure lat/lon geometry, so the match is
exact.

The fixture also pins the land fraction (~18.8 % of the sphere) so an accidental
sign flip or seam error in one continent's inequalities is caught even if the
golden array were ever regenerated.
"""
from pathlib import Path

import numpy as np
import pytest

from jsca.model.land import continents_land_mask

FIXTURE = Path(__file__).parent / "fixtures" / "continents_land_t42.npz"
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="continents land fixture not generated"
)


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


def test_matches_isca_land_generator(fx):
    mine = continents_land_mask(fx["lat"], fx["lon"])
    assert np.array_equal(mine, fx["land_mask"])


def test_values_are_zero_or_one(fx):
    mine = continents_land_mask(fx["lat"], fx["lon"])
    assert set(np.unique(mine)).issubset({0.0, 1.0})


def test_land_fraction_pinned(fx):
    mine = continents_land_mask(fx["lat"], fx["lon"])
    assert mine.mean() == pytest.approx(0.18798828125, abs=1e-9)


def test_subset_is_subset_of_all(fx):
    """Requesting a continent subset yields a strict subset of the full mask."""
    full = continents_land_mask(fx["lat"], fx["lon"], continents=("all",))
    subset = continents_land_mask(fx["lat"], fx["lon"], continents=("AF", "EA"))
    assert np.all(subset <= full)
    assert 0 < subset.sum() < full.sum()


def test_no_land_at_poles(fx):
    """Continents top out near 60 N and reach ~-52 S (South America's tip); the
    polar caps stay ocean."""
    mine = continents_land_mask(fx["lat"], fx["lon"])
    north_cap = fx["lat"] > 62.0
    south_cap = fx["lat"] < -55.0
    assert mine[north_cap].sum() == 0.0
    assert mine[south_cap].sum() == 0.0
