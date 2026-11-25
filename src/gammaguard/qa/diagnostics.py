"""Physics-informed hardware root-cause fault classifier and pre-failure trend forecaster.

Commercial spectroscopy QA suites (Genie 2000 QA, GammaVision QA) only emit pass/fail flags
when FWHM or centroid tolerances are breached. This module analyzes *physical signatures*
across the spectrum to identify which hardware subsystem is failing and estimate Remaining
Useful Life (RUL) before routine QA fails:

  1. Energy-dependent Delta-FWHM^2(E) = da + db*E + dc*E^2 decomposition:
     - Flat da >> 0 (low-energy lines broaden disproportionately) -> cryostat warm-up / microphonics / preamp noise.
     - Growing dc*E^2 + Poisson-significant low-energy peak tailing -> HV bias sag / charge trapping / neutron damage.
  2. High dead time + high-energy (right-side) peak tailing -> pulse pile-up / preamp pole-zero misadjustment.
  3. Calibration drift decomposition: proportional gain drift (amplifier/PMT HV) vs constant offset (ADC zero).
  4. ADC/digitizer single-channel spikes -> stuck bit / DNL fault.
  5. Ambient background window fingerprint: Pb-214/Bi-214 surge vs K-40/Tl-208 -> room radon ingress
     vs broad shielding breach vs discrete spectral contamination.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from ..calibration import EnergyCalibration
from ..peaks.fit import PeakFit
from ..spectrum import Spectrum
from ..stats.control import cusum, ewma, shewhart, standardise
from .baseline import Baseline
from .measure import measure_lines
from .report import QAReport, Status

_ACTIONS: dict[str, str] = {
    "healthy": "Detector operating within baseline physics envelope; continue routine QA schedule.",
    "cryostat_warmup_or_microphonics": (
        "Inspect LN2 dewar level / electromechanical cooler temperature and cold-finger thermal contact; "
        "check mechanical vibration isolation and verify preamp FET electronic noise with a precision pulser."
    ),
    "charge_trapping_or_hv_sag": (
        "Verify detector HV bias supply voltage and leakage current immediately; if bias is nominal on an "
        "HPGe crystal, inspect for neutron radiation damage / incomplete charge collection (consider thermal anneal)."
    ),
    "pileup_or_pole_zero": (
        "Reduce source-to-detector geometry / count rate, verify pile-up rejector (PUR) and live-time clock, "
        "and re-adjust preamplifier pole-zero cancellation on an oscilloscope."
    ),
    "amplifier_or_hv_gain_drift": (
        "Check room temperature stability, spectroscopy amplifier coarse/fine gain, and PMT/detector HV stability; "
        "re-run multi-line energy calibration once thermal equilibrium is confirmed."
    ),
    "adc_baseline_offset_drift": (
        "Inspect ADC zero-level / baseline restorer (BLR) and digital offset settings; verify cable grounding "
        "for ground loops."
    ),
    "adc_digitizer_channel_fault": (
        "Run an MCA sliding-pulse / ramp differential-nonlinearity (DNL) test to isolate stuck ADC bits or "
        "damaged digitizer memory channels."
    ),
    "radon_progeny_ingress": (
        "Verify cave N2 boil-off purge flow and lead-shield door gaskets; ventilate counting room to flush "
        "elevated Rn-222 progeny (Pb-214 / Bi-214)."
    ),
    "shielding_or_broad_background_surge": (
        "Inspect lead shield closure and check nearby rooms/corridors for unshielded sources or storage changes."
    ),
    "spectral_contamination": (
        "Perform wipe test on detector endcap and sample holder to rule out cross-contamination; verify nuclide ID."
    ),
    "missing_or_wrong_source": (
        "Confirm check-source identity, placement geometry, and HV power (reference lines missing or gross shift)."
    ),
}


@dataclass(frozen=True)
class PeakShapeSignature:
    """Poisson-exact low-energy and high-energy tail asymmetry around one fitted peak."""

    line: str
    energy_keV: float
    fwhm_keV: float
    baseline_fwhm_keV: float
    delta_fwhm2_keV2: float
    delta_fwhm2_err_keV2: float
    left_tail_z: float  # excess counts in [mu - 3.5*sigma, mu - 1.2*sigma] over symmetric Gaussian
    right_tail_z: float  # excess counts in [mu + 1.2*sigma, mu + 3.5*sigma]
    asymmetry_z: float  # (left - right) / sqrt(var_left + var_right); positive = low-energy tailing


@dataclass
class HardwareDiagnosis:
    """Structured root-cause diagnosis of a gamma spectrometer measurement."""

    primary_cause: str
    confidence: str  # high | medium | low
    qa_status: str
    likelihoods: dict[str, float]
    observables: dict[str, float | None]
    evidence: list[str]
    recommended_action: str
    peak_shapes: list[PeakShapeSignature] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "primary_cause": self.primary_cause,
            "confidence": self.confidence,
            "qa_status": self.qa_status,
            "likelihoods": {k: round(v, 4) for k, v in self.likelihoods.items()},
            "observables": {
                k: (round(v, 5) if isinstance(v, float) and math.isfinite(v) else None)
                for k, v in self.observables.items()
            },
            "evidence": self.evidence,
            "recommended_action": self.recommended_action,
            "peak_shapes": [
                {
                    "line": p.line,
                    "energy_keV": p.energy_keV,
                    "delta_fwhm2_keV2": round(p.delta_fwhm2_keV2, 4),
                    "left_tail_z": round(p.left_tail_z, 2),
                    "right_tail_z": round(p.right_tail_z, 2),
                    "asymmetry_z": round(p.asymmetry_z, 2),
                }
                for p in self.peak_shapes
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_text(self) -> str:
        lines = [
            f"GammaGuard Hardware Root-Cause Diagnosis: {self.primary_cause} "
            f"(confidence: {self.confidence}, QA: {self.qa_status})",
            f"  Action: {self.recommended_action}",
        ]
        if self.evidence:
            lines.append("  Physical evidence:")
            lines.extend(f"    - {e}" for e in self.evidence)
        ranked = sorted(self.likelihoods.items(), key=lambda kv: -kv[1])[:4]
        lines.append("  Top hypotheses: " + ", ".join(f"{k}={v:.0%}" for k, v in ranked if v >= 0.03))
        return "\n".join(lines)

    def to_markdown(self) -> str:
        lines = [
            f"# GammaGuard Hardware Diagnosis: `{self.primary_cause}`",
            "",
            f"- **Confidence:** {self.confidence}",
            f"- **QA Status:** {self.qa_status}",
            f"- **Recommended Action:** {self.recommended_action}",
            "",
            "## Physical Evidence",
            "",
        ]
        ev_items = self.evidence or ["All physical observables consistent with healthy baseline."]
        lines.extend(f"- {e}" for e in ev_items)
        lines += ["", "## Hypothesis Scores", "", "| Subsystem / Root Cause | Score |", "|---|---|"]
        for k, v in sorted(self.likelihoods.items(), key=lambda kv: -kv[1]):
            if v >= 0.01:
                lines.append(f"| `{k}` | {v:.1%} |")
        return "\n".join(lines) + "\n"


def _peak_shape_signature(
    counts: np.ndarray,
    fit: PeakFit,
    cal: EnergyCalibration,
    label: str,
    energy_keV: float,
    baseline_fwhm_keV: float,
    baseline_fwhm_err_keV: float,
) -> PeakShapeSignature:
    """Measure Poisson-standardized left (low-E) and right (high-E) tail excess relative to symmetric fit."""
    lo, hi = fit.roi
    x = np.arange(lo, hi, dtype=float)
    y = counts[lo:hi].astype(float)
    centre = 0.5 * (lo + hi - 1)
    model = (
        fit.bkg_level
        + fit.bkg_slope * (x - centre)
        + fit.area * (ndtr((x + 0.5 - fit.centroid) / fit.sigma) - ndtr((x - 0.5 - fit.centroid) / fit.sigma))
    )
    if fit.step > 0:
        model = model + fit.step * ndtr((fit.centroid - x) / fit.sigma)

    # Left-side wing [mu - 4.0*sigma, mu - 1.3*sigma] and right-side wing [mu + 1.3*sigma, mu + 4.0*sigma]
    # Note: subtract linear baseline trend symmetrically; compare excess counts vs Poisson noise
    left_mask = (x >= fit.centroid - 4.0 * fit.sigma) & (x <= fit.centroid - 1.3 * fit.sigma)
    right_mask = (x >= fit.centroid + 1.3 * fit.sigma) & (x <= fit.centroid + 4.0 * fit.sigma)

    def _wing_stats(mask: np.ndarray) -> tuple[float, float]:
        if not np.any(mask):
            return 0.0, 1.0
        obs_sum = float(np.sum(y[mask]))
        mod_sum = float(np.sum(np.maximum(model[mask], 0.0)))
        var = max(mod_sum, obs_sum, 1.0)
        return obs_sum - mod_sum, var

    ex_l, var_l = _wing_stats(left_mask)
    ex_r, var_r = _wing_stats(right_mask)
    z_l = ex_l / math.sqrt(var_l)
    z_r = ex_r / math.sqrt(var_r)
    asym_z = (ex_l - ex_r) / math.sqrt(var_l + var_r)

    slope = float(cal.slope(fit.centroid))
    f_keV = fit.fwhm * slope
    fe_keV = fit.fwhm_err * slope
    df2 = f_keV * f_keV - baseline_fwhm_keV * baseline_fwhm_keV
    df2_err = math.hypot(2.0 * f_keV * fe_keV, 2.0 * baseline_fwhm_keV * baseline_fwhm_err_keV)
    return PeakShapeSignature(
        line=label,
        energy_keV=energy_keV,
        fwhm_keV=f_keV,
        baseline_fwhm_keV=baseline_fwhm_keV,
        delta_fwhm2_keV2=df2,
        delta_fwhm2_err_keV2=max(df2_err, 1e-6),
        left_tail_z=z_l,
        right_tail_z=z_r,
        asymmetry_z=asym_z,
    )


def _decompose_delta_fwhm2(shapes: Sequence[PeakShapeSignature]) -> tuple[float, float, float, float]:
    """Fit Delta(FWHM^2)(E) = da + dc * (E/1000)^2 via weighted least squares.

    Returns (da_keV2, da_err, dc_at_1MeV_keV2, dc_err).
    Flat positive ``da`` across energy = electronic noise / cryostat warm-up / microphonics;
    positive ``dc`` growing with E^2 = incomplete charge collection / HV bias sag / neutron damage.
    """
    if len(shapes) < 2:
        if len(shapes) == 1:
            return shapes[0].delta_fwhm2_keV2, shapes[0].delta_fwhm2_err_keV2, 0.0, float("nan")
        return 0.0, float("nan"), 0.0, float("nan")
    e_mev = np.array([s.energy_keV / 1000.0 for s in shapes], dtype=float)
    y = np.array([s.delta_fwhm2_keV2 for s in shapes], dtype=float)
    err = np.array([max(s.delta_fwhm2_err_keV2, 1e-4) for s in shapes], dtype=float)
    w = 1.0 / err
    X = np.column_stack([np.ones_like(e_mev), e_mev * e_mev]) * w[:, None]
    coef, *_ = np.linalg.lstsq(X, y * w, rcond=None)
    cov = np.linalg.pinv(X.T @ X)
    errs = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return float(coef[0]), float(errs[0]), float(coef[1]), float(errs[1])


def diagnose_hardware(
    baseline: Baseline,
    spectrum: Spectrum,
    report: QAReport | None = None,
    *,
    mode: str = "source",
) -> HardwareDiagnosis:
    """Classify the most likely physical hardware or environmental root cause for a measurement."""
    from . import run_qa

    if report is None:
        report = run_qa(baseline, spectrum, mode=mode)

    scores: dict[str, float] = {k: 0.0 for k in _ACTIONS}
    evidence: list[str] = []
    obs: dict[str, float | None] = {}
    shapes: list[PeakShapeSignature] = []

    acq = report.check("acquisition")
    dead_time = acq.metrics.get("dead_time_fraction")
    obs["dead_time_fraction"] = float(dead_time) if dead_time is not None else None

    ch_anom = report.check("channel_anomalies")
    n_anom = int(ch_anom.metrics.get("n_anomalous", 0))
    obs["n_anomalous_channels"] = float(n_anom)

    unexp = report.check("unexpected_peaks")
    n_unexp = int(unexp.metrics.get("n_unexpected", 0))
    obs["n_unexpected_peaks"] = float(n_unexp)

    if mode == "source":
        cal_chk = report.check("calibration")
        res_chk = report.check("resolution")
        active_cal = EnergyCalibration(tuple(report.meta["calibration_used"]))
        widths = {ln.energy: ln.fwhm_keV for ln in baseline.lines}
        opt = baseline.fit
        fits = measure_lines(
            spectrum.counts,
            baseline.calibration,
            list(widths),
            widths.__getitem__,
            search_fwhm=opt.search_fwhm,
            step=opt.step,
            multiplet_fwhm=opt.multiplet_fwhm,
            roi_fwhm=opt.roi_fwhm,
            group_fwhm_keV=lambda e: float(baseline.resolution.fwhm(e)),
        )
        for bl, fit in zip(baseline.lines, fits, strict=True):
            if fit is not None:
                shapes.append(
                    _peak_shape_signature(
                        spectrum.counts, fit, active_cal, bl.label, bl.energy, bl.fwhm_keV, bl.fwhm_err_keV
                    )
                )

        lines_found = int(cal_chk.metrics.get("lines_found", 0))
        lines_exp = int(cal_chk.metrics.get("lines_expected", len(baseline.lines)))
        if lines_found == 0 or lines_found < max(1, lines_exp // 2):
            scores["missing_or_wrong_source"] += 8.0
            evidence.append(f"Only {lines_found}/{lines_exp} reference peaks detected.")

        # 1. Calibration gain vs offset decomposition
        gain_pct = cal_chk.metrics.get("gain_change_pct", float("nan"))
        gain_err = cal_chk.metrics.get("gain_err_pct", float("nan"))
        off_keV = cal_chk.metrics.get("offset_keV", float("nan"))
        off_err = cal_chk.metrics.get("offset_err_keV", float("nan"))
        obs["gain_change_pct"] = gain_pct
        obs["offset_keV"] = off_keV

        gain_ok = math.isfinite(gain_pct) and math.isfinite(gain_err) and gain_err > 0
        off_ok = math.isfinite(off_keV) and math.isfinite(off_err) and off_err > 0
        gain_z = abs(gain_pct / gain_err) if gain_ok else 0.0
        off_z = abs(off_keV / off_err) if off_ok else 0.0

        if cal_chk.status >= Status.WARN:
            if gain_z >= 4.0 and gain_z >= 1.5 * off_z:
                scores["amplifier_or_hv_gain_drift"] += min(9.0, 2.5 + 0.35 * gain_z)
                evidence.append(
                    f"Proportional gain drift {gain_pct:+.4f}% ({gain_z:.1f} sigma) "
                    f"dominates offset ({off_keV:+.3f} keV)."
                )
            elif off_z >= 4.0 and off_z > gain_z:
                scores["adc_baseline_offset_drift"] += min(9.0, 2.5 + 0.35 * off_z)
                evidence.append(
                    f"Constant zero/offset shift {off_keV:+.3f} keV ({off_z:.1f} sigma) "
                    f"dominates gain ({gain_pct:+.4f}%)."
                )
            else:
                scores["amplifier_or_hv_gain_drift"] += 3.0
                scores["adc_baseline_offset_drift"] += 2.0

        # 2. Peak shape tailing & Delta-FWHM^2(E) decomposition
        da, da_err, dc, dc_err = _decompose_delta_fwhm2(shapes)
        mean_left_z = float(np.mean([s.left_tail_z for s in shapes])) if shapes else 0.0
        mean_right_z = float(np.mean([s.right_tail_z for s in shapes])) if shapes else 0.0
        mean_asym_z = float(np.mean([s.asymmetry_z for s in shapes])) if shapes else 0.0
        high_e_left = [s.left_tail_z for s in shapes if s.energy_keV >= 600.0]
        high_e_left_z = float(np.mean(high_e_left)) if high_e_left else mean_left_z

        obs["delta_a_electronic_keV2"] = da
        obs["delta_c_collection_keV2_at_1MeV"] = dc
        obs["mean_left_tail_z"] = mean_left_z
        obs["mean_right_tail_z"] = mean_right_z
        obs["mean_asymmetry_z"] = mean_asym_z

        # Low-energy electronic noise floor check (20-100 keV counts vs baseline)
        low_e_ratio = None
        low_band = next((b for b in baseline.background_bands if b.lo <= 25 and b.hi <= 120), None)
        if low_band is not None and baseline.background_live_time:
            cur_rate = spectrum.band_counts(low_band.lo, low_band.hi, active_cal) / max(spectrum.live_time, 1e-6)
            base_rate = max(low_band.counts / baseline.background_live_time, 1e-3)
            low_e_ratio = cur_rate / base_rate
        obs["low_energy_noise_ratio"] = low_e_ratio

        da_z = da / da_err if math.isfinite(da_err) and da_err > 0 else 0.0
        dc_z = dc / dc_err if math.isfinite(dc_err) and dc_err > 0 else 0.0

        # Check pile-up / pole-zero (high dead time + right-side tailing)
        if (dead_time is not None and dead_time >= baseline.tolerances.dead_time_warn) or mean_right_z >= 3.5:
            pileup_score = 0.0
            if dead_time is not None and dead_time >= baseline.tolerances.dead_time_warn:
                pileup_score += 4.5 + 10.0 * (dead_time - baseline.tolerances.dead_time_warn)
                evidence.append(f"Elevated acquisition dead time ({dead_time:.1%}).")
            if mean_right_z >= 3.0 and mean_right_z > mean_left_z + 1.5:
                pileup_score += min(6.0, 2.0 + 0.6 * mean_right_z)
                evidence.append(
                    f"High-energy (right-side) peak tailing excess ({mean_right_z:+.1f} sigma) "
                    "indicates pile-up or pole-zero undershoot."
                )
            scores["pileup_or_pole_zero"] += pileup_score

        # Check charge trapping / HV bias sag (low-E left-side tailing + energy-growing broadening)
        if high_e_left_z >= 3.0 and mean_asym_z >= 2.2:
            trap_score = min(10.0, 3.5 + 0.55 * high_e_left_z + max(0.0, 0.4 * dc_z))
            scores["charge_trapping_or_hv_sag"] += trap_score
            evidence.append(
                f"Low-energy peak tailing on >=600 keV lines (left wing {high_e_left_z:+.1f} sigma, "
                f"asymmetry {mean_asym_z:+.1f} sigma; Delta-c@1MeV = {dc:+.3f} keV^2) "
                "points to incomplete charge collection / HV sag."
            )

        # Check cryostat warm-up / microphonics / electronic noise (flat da >> 0, symmetric peaks)
        if res_chk.status >= Status.WARN or da_z >= 4.0:
            low_fwhm_ratios = [r["ratio"] for r in res_chk.details if r["energy_keV"] <= 400.0]
            high_fwhm_ratios = [r["ratio"] for r in res_chk.details if r["energy_keV"] >= 1000.0]
            low_gt_high = (
                bool(low_fwhm_ratios and high_fwhm_ratios)
                and float(np.mean(low_fwhm_ratios)) > float(np.mean(high_fwhm_ratios)) + 0.03
            )
            if da_z >= 3.5 and mean_asym_z < 2.5 and (da > max(dc, 0.0) or low_gt_high):
                cryo_score = min(10.0, 3.5 + 0.35 * da_z + (2.0 if low_gt_high else 0.0))
                if low_e_ratio is not None and low_e_ratio > 2.5:
                    cryo_score += 2.0
                scores["cryostat_warmup_or_microphonics"] += cryo_score
                evidence.append(
                    f"Energy-independent broadening Delta-a = {da:+.3f} keV^2 ({da_z:.1f} sigma, symmetric peaks) "
                    "broadens low-energy lines most strongly — cryostat warm-up, microphonics, or preamp noise."
                )
            elif (
                res_chk.status >= Status.WARN
                and scores["charge_trapping_or_hv_sag"] < 2.0
                and scores["pileup_or_pole_zero"] < 2.0
            ):
                # Generic resolution degradation split between electronic noise and collection loss
                if dc_z > da_z:
                    scores["charge_trapping_or_hv_sag"] += 4.0
                else:
                    scores["cryostat_warmup_or_microphonics"] += 4.0
                evidence.append(f"Resolution degraded ({res_chk.summary}).")

    else:
        # Background mode: distinguish radon-progeny ingress vs shielding breach vs contamination
        bkg = report.check("background")
        if bkg.status >= Status.WARN and bkg.details:
            radon_ratios = [
                r["rate_ratio"]
                for r in bkg.details
                if any(tok in r["band"] for tok in ("351.9", "609.3", "1120.3", "1764.5"))
                and r["rate_ratio"] is not None
            ]
            primordial_ratios = [
                r["rate_ratio"]
                for r in bkg.details
                if any(tok in r["band"] for tok in ("1460.8", "2614.5"))
                and r["rate_ratio"] is not None
            ]
            total_ratio = bkg.metrics.get("total_rate_ratio", float("nan"))
            obs["radon_progeny_ratio"] = float(np.mean(radon_ratios)) if radon_ratios else None
            obs["primordial_k40_tl208_ratio"] = float(np.mean(primordial_ratios)) if primordial_ratios else None
            obs["total_background_ratio"] = total_ratio

            r_mean = float(np.mean(radon_ratios)) if radon_ratios else 1.0
            p_mean = float(np.mean(primordial_ratios)) if primordial_ratios else 1.0
            if r_mean >= 1.5 and r_mean > 1.35 * p_mean:
                scores["radon_progeny_ingress"] += min(10.0, 4.0 + 2.5 * (r_mean - p_mean))
                evidence.append(
                    f"Radon progeny windows (Pb-214/Bi-214) surged x{r_mean:.2f} while "
                    f"primordial K-40/Tl-208 stayed x{p_mean:.2f}."
                )
            else:
                scores["shielding_or_broad_background_surge"] += min(10.0, 4.0 + 2.5 * abs(total_ratio - 1.0))
                evidence.append(f"Broad ambient background shifted across bands (total rate ratio x{total_ratio:.2f}).")

    # Cross-mode checks: ADC channel anomalies & unexpected spectral peaks
    if n_anom > 0:
        scores["adc_digitizer_channel_fault"] += min(10.0, 5.5 + 1.5 * n_anom)
        evidence.append(f"Detected {n_anom} hot/dead MCA channel anomaly(ies) ({ch_anom.summary}).")

    if n_unexp > 0 and scores["radon_progeny_ingress"] < 4.0:
        scores["spectral_contamination"] += min(9.0, 4.5 + 1.5 * n_unexp)
        evidence.append(f"Found {n_unexp} unexpected spectral line(s) absent from baseline ({unexp.summary}).")

    # Normalize into probabilities (or report healthy if QA passed and no strong fault signature)
    max_fault = max(v for k, v in scores.items() if k != "healthy")
    if report.status == Status.PASS and max_fault < 3.0:
        scores["healthy"] = 8.0
    else:
        scores["healthy"] = max(0.0, 2.0 - max_fault)

    total_s = sum(scores.values())
    likelihoods = {k: (v / total_s if total_s > 0 else 0.0) for k, v in scores.items()}
    primary = max(likelihoods, key=likelihoods.__getitem__)
    top_p = likelihoods[primary]
    confidence = "high" if top_p >= 0.65 else ("medium" if top_p >= 0.42 else "low")

    return HardwareDiagnosis(
        primary_cause=primary,
        confidence=confidence,
        qa_status=report.status.label,
        likelihoods=likelihoods,
        observables=obs,
        evidence=evidence,
        recommended_action=_ACTIONS[primary],
        peak_shapes=shapes,
    )


@dataclass(frozen=True)
class TrendForecast:
    """Pre-failure early-warning trend summary and Remaining Useful Life (RUL) projection."""

    n_runs: int
    metric: str
    latest_value: float
    drift_rate_per_day: float
    shewhart_alarm_run: int | None
    cusum_alarm_run: int | None
    ewma_alarm_run: int | None
    warn_limit: float
    fail_limit: float
    estimated_days_to_warn: float | None
    estimated_days_to_fail: float | None
    latest_diagnosis: HardwareDiagnosis

    def to_dict(self) -> dict:
        return {
            "n_runs": self.n_runs,
            "metric": self.metric,
            "latest_value": round(self.latest_value, 5),
            "drift_rate_per_day": round(self.drift_rate_per_day, 6),
            "shewhart_alarm_run": self.shewhart_alarm_run,
            "cusum_alarm_run": self.cusum_alarm_run,
            "ewma_alarm_run": self.ewma_alarm_run,
            "warn_limit": self.warn_limit,
            "fail_limit": self.fail_limit,
            "estimated_days_to_warn": (
                round(self.estimated_days_to_warn, 1) if self.estimated_days_to_warn is not None else None
            ),
            "estimated_days_to_fail": (
                round(self.estimated_days_to_fail, 1) if self.estimated_days_to_fail is not None else None
            ),
            "latest_diagnosis": self.latest_diagnosis.to_dict(),
        }

    def to_text(self) -> str:
        lines = [
            f"GammaGuard Pre-Failure Trend Forecast ({self.n_runs} runs, metric={self.metric}):",
            f"  Latest value:   {self.latest_value:.4f} (drift rate {self.drift_rate_per_day:+.5f}/day)",
            f"  Control alarms: Shewhart={self.shewhart_alarm_run}, "
            f"CUSUM={self.cusum_alarm_run}, EWMA={self.ewma_alarm_run}",
            f"  Projected RUL:  to WARN ({self.warn_limit:.3f}) = {self.estimated_days_to_warn} d, "
            f"to FAIL ({self.fail_limit:.3f}) = {self.estimated_days_to_fail} d",
            "",
            self.latest_diagnosis.to_text(),
        ]
        return "\n".join(lines)


def analyze_trend_series(
    baseline: Baseline,
    spectra: Sequence[Spectrum],
    *,
    timestamps_days: Sequence[float] | None = None,
    metric: str = "fwhm_ratio_mean",
    n_reference: int = 5,
    mode: str = "source",
) -> TrendForecast:
    """Run CUSUM/EWMA early-warning control charts and project days-to-breach (RUL)."""
    from . import run_qa

    if len(spectra) < 3:
        raise ValueError("trend analysis requires at least 3 chronological spectra")
    days = (
        np.asarray(timestamps_days, dtype=float)
        if timestamps_days is not None
        else np.arange(len(spectra), dtype=float)
    )
    reports = [run_qa(baseline, sp, mode=mode) for sp in spectra]

    if metric == "fwhm_ratio_mean":
        vals = np.array([r.check("resolution").metrics.get("fwhm_ratio_mean", 1.0) for r in reports], dtype=float)
        warn_lim = baseline.tolerances.fwhm_ratio_warn
        fail_lim = baseline.tolerances.fwhm_ratio_fail
    elif metric == "gain_change_pct":
        vals = np.array([r.check("calibration").metrics.get("gain_change_pct", 0.0) for r in reports], dtype=float)
        # Approximate equivalent % gain at highest reference line
        e_max = max(ln.energy for ln in baseline.lines)
        f_max = max(ln.fwhm_keV for ln in baseline.lines)
        warn_lim = 100.0 * baseline.tolerances.shift_warn_fwhm * f_max / e_max
        fail_lim = 100.0 * baseline.tolerances.shift_fail_fwhm * f_max / e_max
    else:
        raise ValueError(f"unsupported trend metric {metric!r}; choose 'fwhm_ratio_mean' or 'gain_change_pct'")

    ref_n = min(max(3, n_reference), len(vals) - 1)
    try:
        z = standardise(vals, ref_n)
        sh = shewhart(z).alarm_index
        cu = cusum(z).alarm_index
        ew = ewma(z).alarm_index
    except ValueError:
        sh = cu = ew = None

    # Linear drift rate over the recent half (or last >=5 runs) for RUL projection
    tail_n = max(4, len(vals) // 2)
    t_tail = days[-tail_n:]
    v_tail = vals[-tail_n:]
    slope = float(np.polyfit(t_tail, v_tail, 1)[0]) if np.ptp(t_tail) > 0 else 0.0
    latest = float(vals[-1])

    def _rul(limit: float) -> float | None:
        if metric == "fwhm_ratio_mean":
            if latest >= limit:
                return 0.0
            return (limit - latest) / slope if slope > 1e-6 else None
        if abs(latest) >= limit:
            return 0.0
        if abs(slope) <= 1e-6:
            return None
        target = limit if slope > 0 else -limit
        rem = (target - latest) / slope
        return rem if rem >= 0 else None

    latest_diag = diagnose_hardware(baseline, spectra[-1], reports[-1], mode=mode)
    return TrendForecast(
        n_runs=len(spectra),
        metric=metric,
        latest_value=latest,
        drift_rate_per_day=slope,
        shewhart_alarm_run=sh,
        cusum_alarm_run=cu,
        ewma_alarm_run=ew,
        warn_limit=warn_lim,
        fail_limit=fail_lim,
        estimated_days_to_warn=_rul(warn_lim),
        estimated_days_to_fail=_rul(fail_lim),
        latest_diagnosis=latest_diag,
    )
