"""The individual QA checks.

Every check compares a measured spectrum against a `Baseline` and returns a `CheckResult`.
A tolerance breach only counts when it is statistically significant, and a check that cannot
reach a verdict because the data are too coarse says so (WARN) instead of passing silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special
from scipy.ndimage import minimum_filter1d

from ..calibration import EnergyCalibration, weighted_polyfit
from ..isotopes import identify
from ..peaks.finder import find_peak_candidates
from ..peaks.fit import PeakFit, fit_peak
from ..spectrum import Spectrum
from ..stats.poisson import bonferroni_alpha, compare_rates
from .baseline import Baseline
from .report import CheckResult, Status


def _finite(x: float) -> float | None:
    x = float(x)
    return x if math.isfinite(x) else None


def _grade_ratio_interval(lo: float, hi: float, warn: float, fail: float) -> Status:
    """Grade a multiplicative quantity (rate or width ratio) by whether its whole confidence
    interval lies beyond a tolerance. A change that is merely *consistent* with exceeding
    the tolerance does not raise an alarm; one that demonstrably exceeds it does."""
    if lo >= fail or hi <= 1.0 / fail:
        return Status.FAIL
    if lo >= warn or hi <= 1.0 / warn:
        return Status.WARN
    return Status.PASS


def _grade_shift_interval(lo: float, hi: float, warn: float, fail: float) -> Status:
    """Same for an additive shift: the interval must lie entirely outside ``[-tol, +tol]``."""
    nearest = lo if lo > 0 else (-hi if hi < 0 else 0.0)
    if nearest > fail:
        return Status.FAIL
    if nearest > warn:
        return Status.WARN
    return Status.PASS


# ---------------------------------------------------------------------------------------------
def check_acquisition(baseline: Baseline, spectrum: Spectrum) -> CheckResult:
    tol = baseline.tolerances
    status, notes = Status.PASS, []
    metrics = {"total_counts": spectrum.total_counts, "live_time_s": spectrum.live_time}
    if spectrum.n_channels != baseline.n_channels:
        status = Status.FAIL
        notes.append(f"{spectrum.n_channels} channels but baseline has {baseline.n_channels}")
    dt = spectrum.dead_time_fraction
    if dt is None:
        notes.append("real time not recorded, dead time unchecked")
    else:
        metrics["dead_time_fraction"] = dt
        if dt >= tol.dead_time_fail:
            status = Status.FAIL
        elif dt >= tol.dead_time_warn:
            status = max(status, Status.WARN)
        if dt >= tol.dead_time_warn:
            notes.append(f"dead time {dt:.1%}")
    summary = "; ".join(notes) or f"{spectrum.total_counts} counts in {spectrum.live_time:.0f} s live time"
    return CheckResult("acquisition", status, summary, metrics)


# ---------------------------------------------------------------------------------------------
def check_calibration(
    baseline: Baseline, spectrum: Spectrum, fits: list[PeakFit | None]
) -> tuple[CheckResult, EnergyCalibration]:
    """Energy-scale check. Returns the result and the calibration corrected for the measured
    drift (used by the later checks so that small drifts do not masquerade as other faults)."""
    tol, cal = baseline.tolerances, baseline.calibration
    rows, status = [], Status.PASS
    xs, shifts, errs = [], [], []
    worst = None  # (relative severity, row)
    for bl, fit in zip(baseline.lines, fits, strict=True):
        if fit is None:
            status = Status.FAIL
            rows.append({"line": bl.label, "energy_keV": bl.energy, "status": "FAIL", "note": "peak not found"})
            continue
        slope = float(cal.slope(fit.centroid))
        if bl.centroid_ch > 0:  # shift relative to where the healthy baseline measured this line
            shift = slope * (fit.centroid - bl.centroid_ch)
            err = slope * float(np.hypot(fit.centroid_err, bl.centroid_err_ch))
        else:  # baseline files from before centroids were stored: compare with the nominal energy
            shift = float(cal(fit.centroid) - bl.energy)
            err = slope * fit.centroid_err
        z = shift / max(err, 1e-12)
        warn_tol, fail_tol = tol.shift_warn_fwhm * bl.fwhm_keV, tol.shift_fail_fwhm * bl.fwhm_keV
        k = tol.min_significance
        s, note = _grade_shift_interval(shift - k * err, shift + k * err, warn_tol, fail_tol), ""
        if err > tol.precision_warn_frac * warn_tol:
            s, note = max(s, Status.WARN), "centroid precision too coarse to verify tolerance"
        row = {
            "line": bl.label, "energy_keV": bl.energy, "shift_keV": shift, "err_keV": err, "z": z,
            "tol_warn_keV": warn_tol, "tol_fail_keV": fail_tol, "status": s.label, "note": note,
        }
        rows.append(row)
        status = max(status, s)
        xs.append(bl.energy)
        shifts.append(shift)
        errs.append(max(err, 1e-9))
        sev = abs(shift) / warn_tol
        if worst is None or sev > worst[0]:
            worst = (sev, row)

    if len(xs) >= 2:
        coef, cov = weighted_polyfit(xs, shifts, errs, order=1)
        offset, gain = float(coef[0]), float(coef[1])
        offset_err, gain_err = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    elif len(xs) == 1:  # one line cannot separate offset from gain
        offset, gain, offset_err, gain_err = shifts[0], 0.0, errs[0], float("nan")
    else:
        offset = gain = 0.0
        offset_err = gain_err = float("nan")
    corrected = cal.corrected(offset, gain)

    metrics = {
        "lines_found": len(xs), "lines_expected": len(baseline.lines),
        "max_abs_shift_keV": max((abs(s) for s in shifts), default=float("nan")),
        "offset_keV": offset, "offset_err_keV": offset_err,
        "gain_change_pct": 100.0 * gain, "gain_err_pct": 100.0 * gain_err,
    }
    if not xs:
        summary = "no reference peaks found: gross calibration failure or wrong source"
    else:
        missing = len(baseline.lines) - len(xs)
        parts = []
        if missing:
            parts.append(f"{missing} reference peak(s) not found")
        if worst is not None:
            r = worst[1]
            parts.append(f"largest shift vs baseline {r['shift_keV']:+.3f} keV at {r['line']} keV"
                         f" (warn at {r['tol_warn_keV']:.2f})")
        if len(xs) >= 2:
            parts.append(f"fit: offset {offset:+.3f} keV, gain {100 * gain:+.4f} %")
        summary = "; ".join(parts)
    return CheckResult("calibration", status, summary, metrics, rows), corrected


# ---------------------------------------------------------------------------------------------
def check_resolution(
    baseline: Baseline, spectrum: Spectrum, fits: list[PeakFit | None], calibration: EnergyCalibration
) -> CheckResult:
    tol = baseline.tolerances
    rows, ratios, rerrs, status = [], [], [], Status.PASS
    for bl, fit in zip(baseline.lines, fits, strict=True):
        if fit is None:
            continue
        slope = float(calibration.slope(fit.centroid))
        f, fe = fit.fwhm * slope, fit.fwhm_err * slope
        ratio = f / bl.fwhm_keV
        rerr = ratio * math.hypot(fe / f, bl.fwhm_err_keV / bl.fwhm_keV)
        k = tol.min_significance
        s = _grade_ratio_interval(ratio - k * rerr, ratio + k * rerr, tol.fwhm_ratio_warn, tol.fwhm_ratio_fail)
        rows.append({"line": bl.label, "energy_keV": bl.energy, "fwhm_keV": f, "fwhm_err_keV": fe,
                     "baseline_fwhm_keV": bl.fwhm_keV, "ratio": ratio, "ratio_err": rerr, "status": s.label})
        ratios.append(ratio)
        rerrs.append(max(rerr, 1e-9))
        status = max(status, s)
    if not ratios:
        return CheckResult("resolution", Status.SKIPPED, "no reference peaks measured")

    w = 1.0 / np.square(rerrs)
    mean = float(np.sum(w * ratios) / np.sum(w))
    mean_err = float(1.0 / np.sqrt(np.sum(w)))
    k = tol.min_significance
    s_mean = _grade_ratio_interval(mean - k * mean_err, mean + k * mean_err, tol.fwhm_ratio_warn, tol.fwhm_ratio_fail)
    status = max(status, s_mean)
    worst = max(rows, key=lambda r: abs(math.log(r["ratio"])))
    summary = (f"mean FWHM ratio {mean:.3f} +- {mean_err:.3f} vs baseline; "
               f"largest deviation {worst['ratio']:.3f} at {worst['line']} keV "
               f"({worst['fwhm_keV']:.2f} vs {worst['baseline_fwhm_keV']:.2f} keV)")
    metrics = {"fwhm_ratio_mean": mean, "fwhm_ratio_err": mean_err,
               "fwhm_ratio_max": max(ratios), "fwhm_ratio_min": min(ratios)}
    return CheckResult("resolution", status, summary, metrics, rows)


# ---------------------------------------------------------------------------------------------
def check_background(baseline: Baseline, spectrum: Spectrum, calibration: EnergyCalibration) -> CheckResult:
    tol = baseline.tolerances
    if not baseline.background_bands or baseline.background_live_time is None:
        return CheckResult("background", Status.SKIPPED, "baseline has no background reference")
    alpha = bonferroni_alpha(tol.alpha, len(baseline.background_bands))
    rows, status = [], Status.PASS
    for b in baseline.background_bands:
        n_new = spectrum.band_counts(b.lo, b.hi, calibration)
        cmp = compare_rates(n_new, spectrum.live_time, b.counts, baseline.background_live_time,
                            confidence=min(1.0 - 2.0 * alpha, 1.0 - 1e-9))
        s = _grade_ratio_interval(cmp.ci_low, cmp.ci_high, tol.bkg_ratio_warn, tol.bkg_ratio_fail)
        status = max(status, s)
        rows.append({"band": b.label, "counts": n_new, "baseline_counts": b.counts,
                     "rate_ratio": _finite(cmp.ratio), "ci_low": _finite(cmp.ci_low), "ci_high": _finite(cmp.ci_high),
                     "p_value": cmp.p_value, "status": s.label})
    flagged = [r for r in rows if r["status"] != "PASS"]
    total = next((r for r in rows if r["band"].startswith("20-3000")), None)
    metrics = {"bands_tested": len(rows), "bands_flagged": len(flagged),
               "total_rate_ratio": total["rate_ratio"] if total and total["rate_ratio"] is not None else float("nan")}
    if flagged:
        desc = ", ".join(
            f"{r['band']} x{r['rate_ratio']:.2f}" if r["rate_ratio"] is not None else r["band"] for r in flagged[:4]
        )
        summary = f"{len(flagged)}/{len(rows)} bands changed vs baseline: {desc}"
    else:
        summary = f"all {len(rows)} bands consistent with baseline (total rate ratio {metrics['total_rate_ratio']:.2f})"
    return CheckResult("background", status, summary, metrics, rows)


# ---------------------------------------------------------------------------------------------
@dataclass
class _Candidate:
    channel: int
    significance: float
    fit: PeakFit | None
    energy: float
    expected_fwhm_ch: float

    @property
    def peak_like(self) -> bool:
        """A real line has roughly the expected width; a single-channel spike fits far narrower."""
        return self.fit is not None and self.fit.fwhm >= 0.6 * self.expected_fwhm_ch


def find_candidates(baseline: Baseline, spectrum: Spectrum, calibration: EnergyCalibration) -> list[_Candidate]:
    tol = baseline.tolerances
    fwhm_ch = baseline.fwhm_ch(calibration)
    min_ch = int(np.ceil(float(calibration.channel(tol.min_search_keV))))
    out = []
    for c in find_peak_candidates(
        spectrum.counts, fwhm_ch, tol.peak_threshold_sigma, min_channel=min_ch, band=baseline.finder_band
    ):
        width = fwhm_ch(c.channel)
        fit = fit_peak(spectrum.counts, c.channel, width)
        # Only a converged peak may relocate the candidate. A feature that is not Gaussian (a Compton edge,
        # a step) gives a degenerate fit that wanders off by up to 2 FWHM and would no longer match the
        # baseline feature list.
        trusted = fit is not None and fit.significance >= 3.0 and abs(fit.centroid - c.channel) <= 0.75 * width
        centre = fit.centroid if trusted else float(c.channel)
        out.append(_Candidate(c.channel, c.significance, fit, float(calibration(centre)), width))
    return out


def _expected_peaks(baseline: Baseline, mode: str) -> list[float]:
    return baseline.source_peaks if mode == "source" else baseline.background_peaks


def check_unexpected_peaks(
    baseline: Baseline,
    spectrum: Spectrum,
    calibration: EnergyCalibration,
    mode: str = "source",
    candidates: list[_Candidate] | None = None,
) -> CheckResult:
    """Peaks present now that are absent from the healthy baseline spectrum."""
    tol = baseline.tolerances
    if mode == "background" and baseline.background_live_time is None:
        return CheckResult("unexpected_peaks", Status.SKIPPED, "baseline has no background reference")
    expected = _expected_peaks(baseline, mode)
    candidates = find_candidates(baseline, spectrum, calibration) if candidates is None else candidates
    rows = []
    for c in candidates:
        if not c.peak_like:
            continue  # spikes are the channel-anomaly check's business
        if any(abs(c.energy - e0) <= tol.peak_match_fwhm * float(baseline.resolution.fwhm(e0)) for e0 in expected):
            continue
        err_keV = float(calibration.slope(c.fit.centroid) * c.fit.centroid_err)
        hints = identify(c.energy, tolerance=max(1.0, 3.0 * err_keV))
        rows.append({
            "energy_keV": c.energy, "filter_significance": c.significance,
            "net_counts": c.fit.area, "net_counts_err": c.fit.area_err,
            "candidates": ", ".join(h.label for h in hints[:3]) or "none in library",
        })
    status = Status.WARN if rows else Status.PASS
    if rows:
        top = ", ".join(f"{r['energy_keV']:.1f} keV ({r['filter_significance']:.0f} sigma)" for r in rows[:4])
        summary = f"{len(rows)} peak(s) not in the baseline spectrum: {top}"
    else:
        summary = (f"no peaks beyond the {len(expected)} baseline features "
                   f"(threshold {tol.peak_threshold_sigma:g} sigma)")
    return CheckResult("unexpected_peaks", status, summary, {"n_unexpected": len(rows)}, rows)


# ---------------------------------------------------------------------------------------------
_NEIGHBOURS = np.array([1.0, 1.0, 0.0, 1.0, 1.0])  # channels at +-1 and +-2
_P_CENTRE = 1.0 / 5.0


def _neighbour_sum(x: np.ndarray) -> np.ndarray:
    return np.convolve(x, _NEIGHBOURS, mode="same")


def _channel_p_values(obs: np.ndarray, neigh: np.ndarray, rel: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact one-sided p-values (hot, dead) for a channel against the sum ``neigh`` of its four
    neighbours. If all five channels share one mean, ``n | n + S ~ Binomial(n + S, 1/5)``: no
    nuisance parameter and valid at any count level. ``rel`` widens the neighbour sum to allow
    for curvature / differential non-linearity (conservative in both directions)."""
    n = obs.astype(float)
    s_hi, s_lo = neigh * (1.0 + rel), neigh * (1.0 - rel)
    p_hot = np.where(n > 0, special.betainc(np.maximum(n, 1.0), s_hi + 1.0, _P_CENTRE), 1.0)
    p_dead = np.where(s_lo > 0, special.betainc(np.maximum(s_lo, 1e-12), n + 1.0, 1.0 - _P_CENTRE), 1.0)
    return p_hot, p_dead


def _peak_mask(baseline, spectrum, calibration, mode, candidates) -> np.ndarray:
    """Channels where a 5-channel neighbour mean is not a valid model of the local spectrum.

    * known baseline features: a fixed half-width (their amplitude is not stored);
    * detected peaks: grown with the peak's strength. The relative curvature bias of the neighbour
      mean is ~1.25 * |z^2 - 1| / sigma^2 for a Gaussian, so a huge peak needs a much wider mask
      than a small one before the bias drops below the Poisson noise (0.3 sd) of the channel.
    """
    n_ch = spectrum.n_channels
    fwhm_ch = baseline.fwhm_ch(calibration)
    mask = np.zeros(n_ch, dtype=bool)

    def fixed(ch0: float) -> None:
        w = fwhm_ch(ch0)
        # narrow peaks fail the quadratic outright; well-sampled ones only need the core masked
        half = 2.0 * w + 2.0 if w < 12.0 else 0.75 * w + 2.0
        mask[max(int(ch0 - half), 0) : int(ch0 + half) + 1] = True

    for e in _expected_peaks(baseline, mode):
        fixed(float(calibration.channel(e)))
    for c in candidates:
        if not c.peak_like:
            continue
        f = c.fit
        fixed(f.centroid)
        lo, hi = max(int(f.centroid - 20 * f.sigma), 0), min(int(f.centroid + 20 * f.sigma) + 1, n_ch)
        x = np.arange(lo, hi)
        z = (x - f.centroid) / f.sigma
        height = f.area / (f.sigma * np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z**2)
        bias = 1.25 * height * np.abs(z**2 - 1.0) / f.sigma**2
        level = height + max(f.bkg_level, 0.0) + 1.0
        mask[lo:hi] |= bias > 0.3 * np.sqrt(level)
    return mask


def check_channel_anomalies(
    baseline: Baseline,
    spectrum: Spectrum,
    calibration: EnergyCalibration,
    mode: str = "source",
    candidates: list[_Candidate] | None = None,
) -> CheckResult:
    """Single hot or dead channels (digitiser / ADC faults).

    Each channel is tested against its four nearest neighbours with an exact conditional
    binomial test (see `_channel_p_values`) and a Bonferroni correction over all tested
    channels. Flagged channels are replaced by the neighbour mean and the test repeated, so
    that one bad channel does not implicate its neighbours. Channels near peaks are not tested
    (limitation), and a dead channel is only detectable where ~20+ counts per channel are
    expected.
    """
    tol = baseline.tolerances
    candidates = find_candidates(baseline, spectrum, calibration) if candidates is None else candidates
    n_ch = spectrum.n_channels
    lo = max(int(np.ceil(float(calibration.channel(tol.min_search_keV)))), 2)
    tested = np.zeros(n_ch, dtype=bool)
    tested[lo : n_ch - 2] = True
    tested &= ~_peak_mask(baseline, spectrum, calibration, mode, candidates)
    alpha = bonferroni_alpha(tol.alpha, int(tested.sum()))
    rel = 3.0 * tol.channel_rel_floor
    obs = spectrum.counts.astype(float)

    # Peeling: a bad channel corrupts its neighbours' predictions (a hot channel makes its
    # neighbours look dead and vice versa), so in each pass only the most significant channel
    # within +-2 is accepted, replaced by its neighbour mean, and everything is re-tested.
    clean = obs.copy()
    found: dict[int, dict] = {}
    for _ in range(50):
        neigh = _neighbour_sum(clean)
        p_hot, p_dead = _channel_p_values(obs, neigh, rel)
        p_min = np.where(tested, np.minimum(p_hot, p_dead), np.inf)
        cand = np.where(p_min < alpha, p_min, np.inf)
        for ch in found:
            cand[ch] = np.inf
        accept = np.isfinite(cand) & (cand <= minimum_filter1d(cand, size=5))
        if not accept.any():
            break
        for ch in np.flatnonzero(accept):
            hot = bool(p_hot[ch] <= p_dead[ch])
            found[int(ch)] = {"channel": int(ch), "kind": "hot" if hot else "dead", "counts": int(obs[ch]),
                              "expected": float(neigh[ch] / 4.0), "p_value": float(p_min[ch]),
                              "energy_keV": float(calibration(ch))}
        clean[accept] = neigh[accept] / 4.0

    rows = [found[ch] for ch in sorted(found)]
    status = Status.FAIL if rows else Status.PASS
    if rows:
        top = ", ".join(f"ch {r['channel']} {r['kind']} ({r['counts']} vs {r['expected']:.1f})" for r in rows[:4])
        summary = f"{len(rows)} anomalous channel(s): {top}"
    else:
        summary = f"no hot or dead channels among {int(tested.sum())} tested"
    return CheckResult("channel_anomalies", status, summary,
                       {"n_anomalous": len(rows), "n_tested": int(tested.sum())}, rows)
