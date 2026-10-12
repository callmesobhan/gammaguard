import numpy as np
import pytest
from scipy.special import ndtr

from gammaguard.peaks import find_peak_candidates, fit_peak, significance_curve
from gammaguard.qa.measure import measure_line
from gammaguard.simulate import expected_counts, simulate, source_scenario


def _gauss_spectrum(rng, area=5000.0, mu=500.3, sigma=2.0, bkg=20.0, slope=0.0, n=1000):
    x = np.arange(n)
    m = area * (ndtr((x + 0.5 - mu) / sigma) - ndtr((x - 0.5 - mu) / sigma)) + bkg + slope * (x - mu)
    return rng.poisson(m), m


def test_fit_recovers_parameters_within_errors():
    z = {"c": [], "w": [], "a": []}
    for seed in range(60):
        counts, _ = _gauss_spectrum(np.random.default_rng(seed), slope=0.01)
        f = fit_peak(counts, 500, 4.7)
        assert f is not None
        z["c"].append((f.centroid - 500.3) / f.centroid_err)
        z["w"].append((f.sigma - 2.0) / f.sigma_err)
        z["a"].append((f.area - 5000.0) / f.area_err)
    for k, v in z.items():  # unbiased and correctly calibrated errors
        assert abs(np.mean(v)) < 0.4, k
        assert 0.7 < np.std(v) < 1.35, k


def test_fit_handles_near_zero_background():
    """Regression: a ~0.1 counts/channel background used to give a singular start (area_err = 0)."""
    for seed in range(30):
        counts, _ = _gauss_spectrum(np.random.default_rng(seed), area=3500.0, bkg=0.1)
        f = fit_peak(counts, 500, 4.7)
        assert f is not None and f.area_err > 0
        assert f.centroid == pytest.approx(500.3, abs=5 * f.centroid_err)


def test_fit_rejects_empty_region():
    assert fit_peak(np.zeros(1000, dtype=int), 500, 4.7) is None


def test_fit_roi_clipped_at_array_edge():
    counts, _ = _gauss_spectrum(np.random.default_rng(1), mu=12.0, n=200)
    f = fit_peak(counts, 12, 4.7)
    assert f is None or f.roi[0] == 0


def test_finder_finds_injected_lines_and_nothing_else():
    sc = source_scenario()
    spec = simulate(sc, 3)
    cal, res = sc.detector.calibration, sc.detector.resolution
    cands = find_peak_candidates(spec.counts, lambda ch: float(res.fwhm(cal(ch)) / cal.slope(ch)), 5.0, min_channel=100)
    found = np.array([float(cal(c.channel)) for c in cands])
    for e in (302.851, 356.013, 661.657, 1173.228, 1332.492):
        assert np.min(np.abs(found - e)) < 0.6
    assert len(found) <= 9  # 6 intense lines + a few weak Ba-133 lines, no noise peaks


def test_significance_curve_is_flat_noise_for_featureless_poisson_data():
    counts = np.random.default_rng(0).poisson(30.0, 5000)
    sig = significance_curve(counts, 5.0)
    assert abs(sig[50:-50].mean()) < 0.1
    assert 0.8 < sig[50:-50].std() < 1.2
    assert sig.max() < 5.5


def test_measure_line_locates_shifted_peak_and_gives_up_when_absent():
    sc = source_scenario()
    spec = simulate(sc, 5)
    cal = sc.detector.calibration
    f = measure_line(spec.counts, cal, 661.657 + 4.0, 1.5)  # search window centred 4 keV off
    assert f is not None and float(cal(f.centroid)) == pytest.approx(661.657, abs=0.05)
    assert measure_line(spec.counts, cal, 900.0, 1.5) is None
    assert expected_counts(sc).sum() > 0
