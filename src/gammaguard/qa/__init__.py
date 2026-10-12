"""QA pipeline: compare a measurement against a baseline and grade the detector."""

from __future__ import annotations

from ..calibration import EnergyCalibration
from ..spectrum import Spectrum
from .baseline import Baseline, FitOptions, Tolerances
from .checks import (
    check_acquisition,
    check_background,
    check_calibration,
    check_channel_anomalies,
    check_resolution,
    check_unexpected_peaks,
    find_candidates,
)
from .measure import measure_lines
from .profiles import HPGE, NAI, PROFILES, DetectorProfile, get_profile
from .report import CheckResult, QAReport, Status


def run_qa(
    baseline: Baseline,
    spectrum: Spectrum,
    mode: str = "source",
    calibration: EnergyCalibration | None = None,
) -> QAReport:
    """Run all checks that apply to ``mode``.

    ``source``: check-source measurement; calibration, resolution, spectral features.
    ``background``: no source present; background bands and spectral features. Calibration
    cannot be verified without lines, so pass ``calibration`` from the latest source run if
    available (the baseline calibration is used otherwise).
    """
    if mode not in ("source", "background"):
        raise ValueError("mode must be 'source' or 'background'")
    checks = [check_acquisition(baseline, spectrum)]
    meta: dict = {"baseline_created": baseline.created, "live_time_s": spectrum.live_time}

    if mode == "source":
        widths = {ln.energy: ln.fwhm_keV for ln in baseline.lines}
        opt = baseline.fit
        fits = measure_lines(
            spectrum.counts, baseline.calibration, list(widths), widths.__getitem__, search_fwhm=opt.search_fwhm,
            step=opt.step, multiplet_fwhm=opt.multiplet_fwhm, roi_fwhm=opt.roi_fwhm,
            group_fwhm_keV=lambda e: float(baseline.resolution.fwhm(e)),
        )
        cal_result, active = check_calibration(baseline, spectrum, fits)
        checks += [cal_result, check_resolution(baseline, spectrum, fits, active)]
    else:
        active = calibration or baseline.calibration
        checks.append(check_background(baseline, spectrum, active))

    candidates = find_candidates(baseline, spectrum, active)
    checks += [
        check_unexpected_peaks(baseline, spectrum, active, mode, candidates),
        check_channel_anomalies(baseline, spectrum, active, mode, candidates),
    ]
    meta["calibration_used"] = list(active.coeffs)
    return QAReport(mode, checks, meta)


__all__ = [
    "HPGE", "NAI", "PROFILES", "Baseline", "CheckResult", "DetectorProfile", "FitOptions", "QAReport", "Status",
    "Tolerances", "get_profile", "check_acquisition",
    "check_background", "check_calibration", "check_channel_anomalies", "check_resolution",
    "check_unexpected_peaks", "measure_lines", "run_qa",
]
