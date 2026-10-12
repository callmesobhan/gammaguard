import numpy as np
import pytest
from scipy.special import ndtr

from gammaguard import Baseline, Status, run_qa
from gammaguard.calibration import EnergyCalibration
from gammaguard.peaks import fit_peaks
from gammaguard.qa.baseline import Baseline as _Baseline  # noqa: F401
from gammaguard.qa.measure import group_lines, measure_lines
from gammaguard.qa.profiles import NAI, get_profile
from gammaguard.resolution import PowerLawResolution, ResolutionModel, fit_power_law, resolution_from_dict
from gammaguard.simulate import (
    GainDrift,
    HotChannel,
    ResolutionLoss,
    SpuriousPeak,
    background_scenario,
    inject,
    simulate,
    source_scenario,
)


# --- resolution laws ----------------------------------------------------------------------------
def test_power_law_resolution_roundtrip_and_scaling():
    r = PowerLawResolution(1.3, 0.55)
    assert r.fwhm(661.657) == pytest.approx(1.3 * 661.657**0.55)
    assert r.scaled(1.2).fwhm(1000.0) == pytest.approx(1.2 * r.fwhm(1000.0))
    assert resolution_from_dict(r.to_dict()) == r
    old = {"a": 0.5, "b": 0.003, "c": 0.0}  # files written before 'kind' existed are quadrature models
    assert resolution_from_dict(old) == ResolutionModel(0.5, 0.003, 0.0)


def test_fit_power_law_recovers_parameters_and_supports_one_line():
    e = np.array([60.0, 662.0, 1173.0, 1332.0])
    truth = PowerLawResolution(1.4, 0.54)
    fit = fit_power_law(e, truth.fwhm(e), truth.fwhm(e) * 0.01)
    assert (fit.k, fit.p) == pytest.approx((1.4, 0.54), rel=1e-6)
    one = fit_power_law([662.0], [46.0], [0.5], fixed_p=0.55)
    assert one.fwhm(662.0) == pytest.approx(46.0) and one.p == 0.55


# --- multiplet fit -------------------------------------------------------------------------------
def _doublet(rng, c1=400.0, c2=453.0, sigma=9.0, a1=40000.0, a2=30000.0, n=900):
    x = np.arange(n)
    peak = lambda a, mu: a * (ndtr((x + 0.5 - mu) / sigma) - ndtr((x - 0.5 - mu) / sigma))  # noqa: E731
    step = 0.04 * a1 / sigma * ndtr((c1 - x) / sigma) + 0.03 * a2 / sigma * ndtr((c2 - x) / sigma)
    return rng.poisson(peak(a1, c1) + peak(a2, c2) + step + 150.0 - 0.05 * (x - 400))


def test_fit_peaks_separates_a_blended_doublet_with_calibrated_errors():
    z = []
    for seed in range(40):
        counts = _doublet(np.random.default_rng(seed))
        fits = fit_peaks(counts, [402.0, 450.0], [21.2, 21.2], half_width_fwhm=2.0, step=True)
        assert fits is not None and fits[0].group_size == 2
        z += [(fits[0].centroid - 400.0) / fits[0].centroid_err, (fits[1].centroid - 453.0) / fits[1].centroid_err,
              (fits[0].area - 40000.0) / fits[0].area_err, (fits[1].area - 30000.0) / fits[1].area_err]
    z = np.array(z)
    assert abs(z.mean()) < 0.6 and 0.6 < z.std() < 1.5


def test_fit_peaks_returns_results_in_caller_order():
    counts = _doublet(np.random.default_rng(1))
    hi_first = fit_peaks(counts, [450.0, 402.0], [21.2, 21.2], half_width_fwhm=2.0, step=True)
    assert hi_first[0].centroid > hi_first[1].centroid


def test_group_lines_blends_close_pairs_only():
    fw = lambda e: float(PowerLawResolution(1.3, 0.55).fwhm(e))  # noqa: E731
    assert group_lines([59.5, 661.657, 1173.228, 1332.492], fw, 3.0) == [[0], [1], [2, 3]]
    assert group_lines([302.851, 356.013], lambda e: 1.2, 2.5) == [[0], [1]]  # HPGe: resolved


def test_grouping_is_stable_against_noisy_width_estimates():
    """Regression: a pair 2.5 FWHM apart used to be grouped or not depending on the seed width."""
    sc = source_scenario(1800.0, "nai")
    spec = simulate(sc, 1)
    fw_true = lambda e: float(sc.detector.resolution.fwhm(e)) * 0.7  # noqa: E731 - seeds 30 % too narrow
    fits = measure_lines(spec.counts, sc.detector.calibration, [1173.228, 1332.492], fw_true, search_fwhm=2.5,
                         step=True, roi_fwhm=2.0, multiplet_fwhm=3.0,
                         group_fwhm_keV=lambda e: float(NAI.initial_resolution.fwhm(e)))
    assert all(f is not None and f.group_size == 2 for f in fits)


# --- NaI profile, simulated ------------------------------------------------------------------------
@pytest.fixture(scope="module")
def nai_baseline():
    rng = np.random.default_rng(1)
    return Baseline.build(simulate(source_scenario(1800.0, "nai"), rng), simulate(background_scenario(4 * 3600.0, "nai"), rng),
                          profile="nai")


def _run(b, *faults, mode="source", seed=0):
    sc = source_scenario(600.0, "nai") if mode == "source" else background_scenario(3600.0, "nai")
    return run_qa(b, simulate(inject(sc, *faults), seed), mode=mode)


def test_nai_baseline_recovers_the_simulated_detector(nai_baseline):
    b, truth = nai_baseline, source_scenario(1.0, "nai").detector
    assert b.profile == "nai" and isinstance(b.resolution, PowerLawResolution)
    assert b.calibration.coeffs[1] == pytest.approx(3.0, rel=0.01)
    for ln in b.lines:  # the doublet's tied widths are biased low by a few %, everything else is within 3 %
        assert ln.fwhm_keV == pytest.approx(float(truth.resolution.fwhm(ln.energy)), rel=0.10)
    assert b.resolution.p == pytest.approx(0.55, abs=0.05)


def test_nai_baseline_json_roundtrip(nai_baseline, tmp_path):
    nai_baseline.save(tmp_path / "b.json")
    loaded = Baseline.load(tmp_path / "b.json")
    assert loaded.to_dict() == nai_baseline.to_dict() and loaded.fit.step is True
    assert run_qa(loaded, simulate(source_scenario(600.0, "nai"), 3)).status == Status.PASS


@pytest.mark.parametrize("mode", ["source", "background"])
def test_healthy_nai_passes(nai_baseline, mode):
    assert all(_run(nai_baseline, mode=mode, seed=100 + k).status == Status.PASS for k in range(15))


def test_nai_faults_are_caught(nai_baseline):
    assert _run(nai_baseline, GainDrift(0.02)).check("calibration").status == Status.FAIL
    assert _run(nai_baseline, GainDrift(0.002)).check("calibration").status == Status.PASS
    assert _run(nai_baseline, ResolutionLoss(1.3)).check("resolution").status == Status.FAIL
    assert _run(nai_baseline, SpuriousPeak(2200.0, 1.5)).check("unexpected_peaks").status == Status.WARN


def test_hot_channel_in_search_window_does_not_mislead_calibration(nai_baseline):
    """Regression: a strong single-channel spike next to the Co-60 doublet used to win the peak search."""
    rep = _run(nai_baseline, HotChannel(500, 2.0))
    assert rep.check("calibration").status == Status.PASS
    assert rep.check("calibration").metrics["lines_found"] == 4


def test_single_line_baseline_for_a_daily_cs137_check():
    rng = np.random.default_rng(3)
    b = Baseline.build(simulate(source_scenario(1800.0, "nai"), rng), profile="nai", reference_energies=[661.657])
    assert len(b.lines) == 1 and b.resolution.p == NAI.power_law_exponent
    healthy = run_qa(b, simulate(source_scenario(600.0, "nai"), 5))
    drifted = run_qa(b, simulate(inject(source_scenario(600.0, "nai"), GainDrift(0.03)), 6))
    assert healthy.check("calibration").status == Status.PASS
    assert drifted.check("calibration").status == Status.FAIL


def test_unknown_profile_rejected():
    with pytest.raises(ValueError, match="unknown detector profile"):
        get_profile("scintillator-9000")


# --- drift is measured against the baseline, not against nominal energies -----------------------------
def test_baseline_relative_shift_ignores_integral_nonlinearity():
    """Regression for real data: a detector with INL leaves ~0.15 keV residuals against nominal line
    energies after a linear calibration. Those are properties of the baseline, not drift."""
    base_sc = source_scenario(1800.0)
    det = base_sc.detector
    from dataclasses import replace
    inl = replace(det, actual_calibration=EnergyCalibration((0.0, 0.4, 1.0e-7)))
    sc = lambda t: replace(base_sc, detector=inl, live_time=t)  # noqa: E731
    b = Baseline.build(simulate(sc(1800.0), 1))
    rep = run_qa(b, simulate(sc(600.0), 2))
    cal = rep.check("calibration")
    assert cal.status == Status.PASS
    assert cal.metrics["max_abs_shift_keV"] < 0.05  # versus the ~0.15 keV INL residuals vs nominal energies
