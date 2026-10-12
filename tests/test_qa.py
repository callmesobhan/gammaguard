import numpy as np
import pytest

from gammaguard import Baseline, Status, Tolerances, run_qa
from gammaguard.simulate import (
    BackgroundScale,
    DeadChannel,
    DeadTime,
    GainDrift,
    HotChannel,
    NoiseBurst,
    RadonSurge,
    ResolutionLoss,
    SpuriousPeak,
    ZeroShift,
    background_scenario,
    inject,
    simulate,
    source_scenario,
)


def _run(baseline, *faults, mode="source", seed=0):
    sc = source_scenario() if mode == "source" else background_scenario()
    return run_qa(baseline, simulate(inject(sc, *faults), seed), mode=mode)


# --- baseline ---------------------------------------------------------------------------------
def test_baseline_matches_simulated_truth(baseline):
    truth = source_scenario().detector
    assert baseline.calibration.coeffs[1] == pytest.approx(truth.calibration.coeffs[1], rel=1e-5)
    for ln in baseline.lines:
        assert ln.fwhm_keV == pytest.approx(float(truth.resolution.fwhm(ln.energy)), rel=0.03)
    assert len(baseline.lines) == 5
    assert baseline.background_bands and baseline.source_peaks and baseline.background_peaks


def test_baseline_json_roundtrip(baseline, tmp_path):
    path = tmp_path / "baseline.json"
    baseline.save(path)
    loaded = Baseline.load(path)
    assert loaded.to_dict() == baseline.to_dict()
    assert run_qa(loaded, simulate(source_scenario(), 1)).status == Status.PASS


def test_baseline_needs_calibrated_spectrum():
    spec = simulate(source_scenario(), 1)
    spec.calibration = None
    with pytest.raises(ValueError, match="calibration"):
        Baseline.build(spec)


# --- healthy detector --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["source", "background"])
def test_healthy_detector_passes(baseline, mode):
    statuses = [_run(baseline, mode=mode, seed=100 + k).status for k in range(25)]
    assert all(s == Status.PASS for s in statuses), statuses


def test_regression_valid_peak_not_reported_missing():
    """Seed from a real false FAIL: the 1173 keV peak sat in a ~0.1 count/channel background."""
    r_s, r_b, r_t = (np.random.default_rng(s) for s in np.random.SeedSequence(1002).spawn(3))
    b = Baseline.build(simulate(source_scenario(1800.0), r_s), simulate(background_scenario(4 * 3600.0), r_b))
    report = run_qa(b, simulate(source_scenario(), r_t))
    assert report.check("calibration").metrics["lines_found"] == 5
    assert report.status == Status.PASS


# --- source-mode faults ---------------------------------------------------------------------------
def test_gain_drift_above_tolerance_is_warned_and_quantified(baseline):
    report = _run(baseline, GainDrift(3e-4))
    cal = report.check("calibration")
    assert cal.status == Status.WARN
    assert cal.metrics["gain_change_pct"] == pytest.approx(0.03, abs=0.005)
    assert abs(cal.metrics["offset_keV"]) < 0.05


def test_large_gain_drift_fails_but_other_checks_stay_quiet(baseline):
    report = _run(baseline, GainDrift(1e-3))
    assert report.check("calibration").status == Status.FAIL
    # drift is corrected for before the feature checks, so it must not look like new peaks
    assert report.check("unexpected_peaks").status == Status.PASS
    assert report.check("resolution").status == Status.PASS


def test_gain_drift_below_tolerance_is_not_an_alarm(baseline):
    assert _run(baseline, GainDrift(1e-4)).check("calibration").status == Status.PASS


def test_zero_shift_is_attributed_to_offset(baseline):
    cal = _run(baseline, ZeroShift(1.0)).check("calibration")
    assert cal.status == Status.FAIL
    assert cal.metrics["offset_keV"] == pytest.approx(1.0, abs=0.05)
    assert abs(cal.metrics["gain_change_pct"]) < 0.005


def test_resolution_loss(baseline):
    assert _run(baseline, ResolutionLoss(1.25)).check("resolution").status == Status.FAIL
    assert _run(baseline, ResolutionLoss(1.04)).check("resolution").status == Status.PASS
    res = _run(baseline, ResolutionLoss(1.25)).check("resolution")
    assert res.metrics["fwhm_ratio_mean"] == pytest.approx(1.25, abs=0.03)


def test_unexpected_peak_is_found_and_identified(baseline):
    report = _run(baseline, SpuriousPeak(1274.0, 1.0))  # near the Na-22 line
    chk = report.check("unexpected_peaks")
    assert chk.status == Status.WARN
    row = chk.details[0]
    assert row["energy_keV"] == pytest.approx(1274.0, abs=0.5)
    assert "Na-22" in row["candidates"]


def test_hot_and_dead_channels(baseline):
    hot = _run(baseline, HotChannel(2200, 0.5)).check("channel_anomalies")
    assert hot.status == Status.FAIL
    assert [(d["channel"], d["kind"]) for d in hot.details] == [(2200, "hot")]
    dead = _run(baseline, DeadChannel(1000)).check("channel_anomalies")
    assert [(d["channel"], d["kind"]) for d in dead.details] == [(1000, "dead")]


def test_hot_channel_does_not_implicate_its_neighbours(baseline):
    chk = _run(baseline, HotChannel(2200, 2.0)).check("channel_anomalies")  # huge spike
    assert [d["channel"] for d in chk.details] == [2200]


def test_hot_and_dead_channels_together(baseline):
    faults = [HotChannel(2200, 0.5), HotChannel(4500, 0.3), DeadChannel(1000), DeadChannel(1100)]
    chk = _run(baseline, *faults).check("channel_anomalies")
    assert {(d["channel"], d["kind"]) for d in chk.details} == {
        (2200, "hot"), (4500, "hot"), (1000, "dead"), (1100, "dead")}


def test_dead_time_fails_acquisition(baseline):
    assert _run(baseline, DeadTime(0.35)).check("acquisition").status == Status.FAIL
    assert _run(baseline, DeadTime(0.15)).check("acquisition").status == Status.WARN


def test_missing_reference_peaks_fail_calibration(baseline):
    spec = simulate(background_scenario(600.0), 3)  # no source in the cave
    spec.calibration = baseline.calibration
    cal = run_qa(baseline, spec, mode="source").check("calibration")
    assert cal.status == Status.FAIL and "no reference peaks" in cal.summary
    assert cal.metrics["lines_found"] == 0


def test_partially_missing_reference_peaks_fail_calibration(baseline):
    spec = simulate(source_scenario(), 3)
    counts = spec.counts.copy()
    ch = int(round(float(baseline.calibration.channel(661.657))))
    counts[ch - 40 : ch + 41] = np.median(counts[ch + 60 : ch + 100])  # erase the Cs-137 line
    from gammaguard import Spectrum
    cal = run_qa(baseline, Spectrum(counts, spec.live_time, spec.real_time, spec.calibration)).check("calibration")
    assert cal.status == Status.FAIL and "1 reference peak(s) not found" in cal.summary
    assert cal.metrics["lines_found"] == 4


def test_channel_count_mismatch_fails_acquisition(baseline):
    spec = simulate(source_scenario(), 1)
    from gammaguard import Spectrum
    short = Spectrum(spec.counts[:4096], spec.live_time, spec.real_time, spec.calibration)
    assert run_qa(baseline, short).check("acquisition").status == Status.FAIL


# --- background-mode faults -------------------------------------------------------------------------
def test_background_increase_is_detected_and_ordered_by_severity(baseline):
    assert _run(baseline, BackgroundScale(1.1), mode="background").check("background").status == Status.PASS
    assert _run(baseline, BackgroundScale(1.5), mode="background").check("background").status == Status.WARN
    assert _run(baseline, BackgroundScale(2.5), mode="background").check("background").status == Status.FAIL


def test_radon_and_noise_flag_the_right_bands(baseline):
    radon = _run(baseline, RadonSurge(3.0), mode="background").check("background")
    assert radon.status == Status.FAIL
    flagged = {d["band"] for d in radon.details if d["status"] != "PASS"}
    assert "609.3 keV line" in flagged
    noise = _run(baseline, NoiseBurst(0.5), mode="background").check("background")
    assert noise.status == Status.FAIL
    assert any(d["band"] == "20-100 keV" and d["status"] != "PASS" for d in noise.details)


def test_background_check_skipped_without_reference():
    rng = np.random.default_rng(0)
    b = Baseline.build(simulate(source_scenario(1800.0), rng))
    report = run_qa(b, simulate(background_scenario(), 1), mode="background")
    assert report.check("background").status == Status.SKIPPED
    assert report.status == Status.PASS  # skipped checks never block


# --- tolerances and reporting --------------------------------------------------------------------------
def test_tolerances_are_configurable():
    rng = np.random.default_rng(0)
    strict = Tolerances(shift_warn_fwhm=0.01, shift_fail_fwhm=0.02)
    b = Baseline.build(simulate(source_scenario(1800.0), rng), tolerances=strict)
    assert run_qa(b, simulate(inject(source_scenario(), GainDrift(1e-4)), 1)).check("calibration").status == Status.FAIL


def test_report_formats_and_exit_codes(baseline):
    ok, bad = _run(baseline), _run(baseline, GainDrift(1e-3), HotChannel(2200, 0.5))
    assert (ok.exit_code, bad.exit_code) == (0, 2)
    assert "FAIL" in bad.to_text() and "calibration" in bad.to_markdown()
    import json
    d = json.loads(bad.to_json())
    assert d["status"] == "FAIL" and {c["name"] for c in d["checks"]} >= {"calibration", "channel_anomalies"}
    with pytest.raises(KeyError):
        bad.check("nope")
