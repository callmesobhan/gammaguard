import numpy as np
import pytest

from gammaguard.calibration import EnergyCalibration, fit_calibration, weighted_polyfit
from gammaguard.resolution import FWHM_PER_SIGMA, ResolutionModel, fit_resolution


def test_linear_calibration_roundtrip():
    cal = EnergyCalibration((1.5, 0.4))
    assert cal(100) == pytest.approx(41.5)
    assert cal.channel(41.5) == pytest.approx(100)
    assert cal.slope(5) == pytest.approx(0.4)


def test_quadratic_inverse_by_newton():
    cal = EnergyCalibration((0.5, 0.4, 2e-8))
    ch = np.array([10.0, 1000.0, 7000.0])
    assert cal.channel(cal(ch)) == pytest.approx(ch, rel=1e-9)


def test_invalid_calibration_rejected():
    with pytest.raises(ValueError):
        EnergyCalibration((1.0,))
    with pytest.raises(ValueError):
        EnergyCalibration((0.0, -0.4))


def test_corrected_undoes_reading_error():
    true_cal = EnergyCalibration((0.2, 0.4))
    # a drifted detector read through the old calibration: E_read = (1 + g) E + z
    g, z = 3e-4, 0.15
    read = EnergyCalibration((true_cal.coeffs[0] * (1 + g) + z, true_cal.coeffs[1] * (1 + g)))
    fixed = read.corrected(offset_keV=z, gain_fraction=g)
    ch = np.array([100.0, 3000.0])
    assert fixed(ch) == pytest.approx(true_cal(ch), abs=1e-6)


def test_fit_calibration_recovers_coefficients():
    ch = np.array([757.0, 890.0, 1654.0, 2933.0, 3331.0])
    cal = EnergyCalibration((0.3, 0.4001))
    fitted = fit_calibration(ch, cal(ch), channel_sigma=np.full(5, 0.02))
    assert fitted.coeffs == pytest.approx(cal.coeffs, rel=1e-9)


def test_weighted_polyfit_covariance_scales_with_sigma():
    x = np.linspace(0, 10, 20)
    _, cov1 = weighted_polyfit(x, 2 * x, np.full(20, 1.0))
    _, cov2 = weighted_polyfit(x, 2 * x, np.full(20, 2.0))
    assert cov2 == pytest.approx(4 * cov1)


def test_resolution_model_scaling_and_sigma():
    m = ResolutionModel(0.488, 0.002636)
    assert m.fwhm(1332.5) == pytest.approx(2.0, abs=0.01)
    assert m.scaled(1.2).fwhm(661.7) == pytest.approx(1.2 * m.fwhm(661.7))
    assert m.sigma(100.0) * FWHM_PER_SIGMA == pytest.approx(m.fwhm(100.0))


def test_fit_resolution_recovers_model():
    m = ResolutionModel(0.488, 0.002636)
    e = np.array([300.0, 356.0, 662.0, 1173.0, 1332.0])
    fitted = fit_resolution(e, m.fwhm(e), fwhm_err=np.full(5, 0.01))
    assert fitted.fwhm(e) == pytest.approx(m.fwhm(e), rel=1e-4)
    assert fitted.a >= 0 and fitted.b >= 0
