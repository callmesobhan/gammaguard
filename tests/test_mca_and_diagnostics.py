"""Tests for raw vendor MCA I/O (.spe/.chn/.n42/.cnf) and physics-informed hardware root-cause diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gammaguard.cli import main
from gammaguard.io import read_spectrum, write_spectrum
from gammaguard.qa import Baseline, analyze_trend_series, diagnose_hardware, run_qa
from gammaguard.simulate import (
    BackgroundScale,
    ChargeTrapping,
    CryostatWarmup,
    GainDrift,
    HotChannel,
    PileUp,
    RadonSurge,
    SpuriousPeak,
    ZeroShift,
    background_scenario,
    inject,
    simulate,
    source_scenario,
)


@pytest.fixture(scope="module")
def hpge_baseline() -> Baseline:
    rng = np.random.default_rng(101)
    src = simulate(source_scenario(1800.0, "hpge"), rng)
    bkg = simulate(background_scenario(14400.0, "hpge"), rng)
    return Baseline.build(src, bkg, profile="hpge")


@pytest.mark.parametrize("ext", [".spe", ".chn", ".n42", ".cnf"])
def test_vendor_mca_formats_roundtrip_and_pass_qa(tmp_path: Path, hpge_baseline: Baseline, ext: str):
    spec = simulate(source_scenario(600.0, "hpge"), np.random.default_rng(7))
    out = tmp_path / f"check_source{ext}"
    write_spectrum(spec, out)
    loaded = read_spectrum(out)

    assert loaded.n_channels == spec.n_channels
    assert np.array_equal(loaded.counts, spec.counts)
    assert loaded.live_time == pytest.approx(spec.live_time, rel=1e-4)
    assert loaded.real_time == pytest.approx(spec.real_time, rel=1e-4)
    assert loaded.calibration is not None
    assert list(loaded.calibration.coeffs[:2]) == pytest.approx(list(spec.calibration.coeffs[:2]), rel=1e-3)

    report = run_qa(hpge_baseline, loaded)
    assert report.status.label == "PASS"
    diag = diagnose_hardware(hpge_baseline, loaded, report)
    assert diag.primary_cause == "healthy"


def test_diagnose_separates_cryostat_warmup_from_charge_trapping(hpge_baseline: Baseline):
    # Cryostat warm-up / microphonics: raises energy-independent Delta-a (symmetric broadening)
    cryo_spec = simulate(
        inject(source_scenario(1800.0, "hpge"), CryostatWarmup(delta_a_keV2=1.4)),
        np.random.default_rng(11),
    )
    cryo_diag = diagnose_hardware(hpge_baseline, cryo_spec)
    assert cryo_diag.primary_cause == "cryostat_warmup_or_microphonics"
    assert cryo_diag.observables["delta_a_electronic_keV2"] > 0.6

    # Neutron damage / HV bias sag: energy-growing low-energy tailing + c*E^2 incomplete collection
    trap_spec = simulate(
        inject(source_scenario(1800.0, "hpge"), ChargeTrapping(tail_fraction=0.30)),
        np.random.default_rng(12),
    )
    trap_diag = diagnose_hardware(hpge_baseline, trap_spec)
    assert trap_diag.primary_cause == "charge_trapping_or_hv_sag"
    assert trap_diag.observables["mean_left_tail_z"] > 3.0


def test_diagnose_identifies_pileup_gain_offset_adc_radon_shielding_and_contamination(hpge_baseline: Baseline):
    rng = np.random.default_rng(20)

    pileup_spec = simulate(inject(source_scenario(1200.0, "hpge"), PileUp(0.28, 0.25)), rng)
    assert diagnose_hardware(hpge_baseline, pileup_spec).primary_cause == "pileup_or_pole_zero"

    gain_spec = simulate(inject(source_scenario(900.0, "hpge"), GainDrift(0.0008)), rng)
    assert diagnose_hardware(hpge_baseline, gain_spec).primary_cause == "amplifier_or_hv_gain_drift"

    zero_spec = simulate(inject(source_scenario(900.0, "hpge"), ZeroShift(0.8)), rng)
    assert diagnose_hardware(hpge_baseline, zero_spec).primary_cause == "adc_baseline_offset_drift"

    adc_spec = simulate(inject(source_scenario(900.0, "hpge"), HotChannel(2200, 1.0)), rng)
    assert diagnose_hardware(hpge_baseline, adc_spec).primary_cause == "adc_digitizer_channel_fault"

    contam_spec = simulate(inject(source_scenario(900.0, "hpge"), SpuriousPeak(800.0, 2.0)), rng)
    assert diagnose_hardware(hpge_baseline, contam_spec).primary_cause == "spectral_contamination"

    radon_spec = simulate(inject(background_scenario(7200.0, "hpge"), RadonSurge(4.0)), rng)
    assert diagnose_hardware(hpge_baseline, radon_spec, mode="background").primary_cause == "radon_progeny_ingress"

    shield_spec = simulate(inject(background_scenario(7200.0, "hpge"), BackgroundScale(2.5)), rng)
    assert (
        diagnose_hardware(hpge_baseline, shield_spec, mode="background").primary_cause
        == "shielding_or_broad_background_surge"
    )


def test_pre_failure_trend_forecast_detects_slow_warmup_and_projects_rul(hpge_baseline: Baseline):
    rng = np.random.default_rng(33)
    spectra = []
    # 6 healthy days, then slow cryostat warm-up ramp (still below single-run FAIL threshold!)
    for day in range(14):
        da = 0.0 if day < 6 else 0.06 * (day - 5)
        sc = source_scenario(900.0, "hpge")
        if da > 0:
            sc = inject(sc, CryostatWarmup(delta_a_keV2=da))
        spectra.append(simulate(sc, rng))

    fc = analyze_trend_series(hpge_baseline, spectra, metric="fwhm_ratio_mean", n_reference=5)
    assert fc.cusum_alarm_run is not None
    assert fc.ewma_alarm_run is not None
    assert fc.drift_rate_per_day > 0.0
    assert fc.estimated_days_to_fail is not None and fc.estimated_days_to_fail > 0.0


def test_cli_diagnose_and_trend_on_vendor_spe_files(tmp_path: Path, hpge_baseline: Baseline, capsys):
    base_path = tmp_path / "base.json"
    hpge_baseline.save(base_path)

    spe_path = tmp_path / "fault.spe"
    spec = simulate(inject(source_scenario(1200.0, "hpge"), ChargeTrapping(0.30)), np.random.default_rng(55))
    write_spectrum(spec, spe_path)

    rc = main(["diagnose", "--baseline", str(base_path), "--format", "json", str(spe_path)])
    assert rc in (0, 1, 2)
    payload = json.loads(capsys.readouterr().out)
    assert payload["primary_cause"] == "charge_trapping_or_hv_sag"

    # Run trend CLI over 6 .chn files
    chn_paths = []
    rng = np.random.default_rng(77)
    for i in range(6):
        p = tmp_path / f"day_{i:02d}.chn"
        sc = inject(source_scenario(600.0, "hpge"), GainDrift(0.00004 * i)) if i >= 3 else source_scenario(600.0, "hpge")
        write_spectrum(simulate(sc, rng), p)
        chn_paths.append(str(p))

    rc_tr = main(["trend", "--baseline", str(base_path), "--metric", "gain_change_pct", "--format", "json", *chn_paths])
    assert rc_tr == 0
    tr_out = json.loads(capsys.readouterr().out)
    assert tr_out["n_runs"] == 6
    assert "latest_diagnosis" in tr_out
