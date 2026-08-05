"""Tier-1 tests for vert_advection against real-Fortran fixtures.

Fixtures come from ``fortran_instrumentation/dump_vert_advection_reference.F90``,
which compiles Isca's actual ``vert_advection.F90`` unmodified (fms_mod stubbed;
mpp_mod a serial no-op). Regeneration recipe in that file's header.

Fortran storage is ``(lon, lat, level)`` with level LAST — exactly the port's
``(..., K)`` column layout — so no axis move is needed; the ``(lon, lat)`` axes
are simply leading batch dimensions. Every ported scheme is checked in both
``ADVECTIVE_FORM`` and ``FLUX_FORM``.
"""

from functools import partial
from pathlib import Path

import jax
import numpy as np
import pytest

import jsca  # noqa: F401  (enables x64)
from jsca.dycore import (
    ADVECTIVE_FORM,
    FINITE_VOLUME_LINEAR,
    FLUX_FORM,
    FOURTH_CENTERED,
    FOURTH_CENTERED_WTS,
    SECOND_CENTERED,
    SECOND_CENTERED_WTS,
    vert_advection,
)

FIXTURE = Path(__file__).parent / "fixtures" / "vert_advection_reference.npz"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="vert_advection fixtures not generated"
)

SCHEMES = {
    "2c": SECOND_CENTERED,
    "2cw": SECOND_CENTERED_WTS,
    "4c": FOURTH_CENTERED,
    "4cw": FOURTH_CENTERED_WTS,
    "fvl": FINITE_VOLUME_LINEAR,
}
FORMS = {"adv": ADVECTIVE_FORM, "flux": FLUX_FORM}


@pytest.fixture(scope="module")
def fx():
    return np.load(FIXTURE)


@pytest.mark.parametrize("scheme_tag", list(SCHEMES))
@pytest.mark.parametrize("form_tag", list(FORMS))
def test_vert_advection(fx, scheme_tag, form_tag):
    dt = float(fx["va_meta"][3])
    out = vert_advection(
        dt, fx["va_w"], fx["va_dz"], fx["va_r"],
        scheme=SCHEMES[scheme_tag], form=FORMS[form_tag],
    )
    ref = fx[f"va_{scheme_tag}_{form_tag}"]
    np.testing.assert_allclose(np.asarray(out), ref, rtol=1e-14, atol=1e-16)


def test_jit_and_1d_column(fx):
    """jit (scheme/form static) matches eager; a bare 1-D column matches batch [0, 0]."""
    dt = float(fx["va_meta"][3])
    w, dz, r = fx["va_w"], fx["va_dz"], fx["va_r"]
    jitted = jax.jit(partial(vert_advection, scheme=SECOND_CENTERED, form=ADVECTIVE_FORM))
    out = np.asarray(jitted(dt, w, dz, r))
    np.testing.assert_allclose(out, fx["va_2c_adv"], rtol=1e-14, atol=1e-16)
    out1d = np.asarray(
        vert_advection(dt, w[0, 0], dz[0, 0], r[0, 0], scheme=SECOND_CENTERED, form=ADVECTIVE_FORM)
    )
    np.testing.assert_allclose(out1d, fx["va_2c_adv"][0, 0], rtol=1e-14, atol=1e-16)
