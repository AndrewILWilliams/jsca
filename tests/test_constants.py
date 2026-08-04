import numpy as np

from jsca import constants


def test_derived_relationships_match_fortran():
    np.testing.assert_allclose(constants.CP_AIR, 287.04 / (2.0 / 7.0), rtol=1e-15)
    np.testing.assert_allclose(constants.CP_AIR, 1004.64, rtol=1e-12)
    assert constants.HLS == constants.HLV + constants.HLF


def test_isca_specific_values_not_silently_corrected():
    # These differ from common textbook values on purpose (they match Isca).
    assert constants.RADIUS == 6376.0e3  # not 6371e3
    assert constants.GRAV == 9.80  # not 9.81
    assert constants.STEFAN == 5.6734e-8  # not 5.670374e-8
    assert constants.OMEGA == 7.2921150e-5
