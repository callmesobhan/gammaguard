"""The baseline: what a healthy detector looks like, captured from known-good measurements."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..calibration import EnergyCalibration, fit_calibration
from ..isotopes import all_lines
from ..peaks.finder import find_peak_candidates
from ..resolution import (
    Resolution,
    ResolutionModel,
    fit_power_law,
    fit_resolution,
    resolution_from_dict,
)
from ..spectrum import Spectrum
from .measure import measure_lines

# Broad background bands (keV) plus narrow windows on the usual ambient lines.
BACKGROUND_BANDS: tuple[tuple[float, float], ...] = (
    (20, 100), (100, 300), (300, 600), (600, 1000), (1000, 1500), (1500, 3000),
)
BACKGROUND_WINDOW_LINES: tuple[float, ...] = (351.93, 511.0, 609.31, 1120.29, 1460.82, 1764.49, 2614.51)


@dataclass
class Tolerances:
    """Acceptance criteria. An alarm needs the whole confidence interval beyond the tolerance, so
    low-statistics measurements cannot raise alarms through noise alone."""

    # Calibration: reading error of a line, as a fraction of the baseline FWHM at that line.
    shift_warn_fwhm: float = 0.10
    shift_fail_fwhm: float = 0.25
    precision_warn_frac: float = 0.5  # centroid error above this fraction of warn tolerance
    # Resolution: measured / baseline FWHM.
    fwhm_ratio_warn: float = 1.08
    fwhm_ratio_fail: float = 1.15
    # Background: rate / baseline rate (also applied inverted for decreases).
    bkg_ratio_warn: float = 1.25
    bkg_ratio_fail: float = 1.60
    # Shared.
    min_significance: float = 3.0  # half-width of the confidence interval, in sigma
    alpha: float = 1e-3  # family-wise false-alarm probability per statistical family
    # Spectral features.
    peak_threshold_sigma: float = 5.5
    baseline_peak_threshold: float = 3.0
    peak_match_fwhm: float = 2.0
    min_search_keV: float = 30.0
    channel_rel_floor: float = 0.03  # allowance for differential non-linearity / model error
    # Acquisition.
    dead_time_warn: float = 0.10
    dead_time_fail: float = 0.30


@dataclass
class FitOptions:
    """How reference peaks are located and fitted (set by the detector profile)."""

    step: bool = False  # add a Compton-step term under each peak
    roi_fwhm: float = 4.0  # half-width of the fit region, in FWHM
    search_fwhm: float = 10.0  # how far from the expected position to look for a peak, in FWHM
    multiplet_fwhm: float = 2.5  # lines closer than this many FWHM are fitted together


@dataclass
class BaselineLine:
    energy: float
    label: str
    fwhm_keV: float
    fwhm_err_keV: float
    area_rate: float  # net counts per live second (informational)
    # Where the line sat in the healthy baseline spectrum (channels). Drift is measured against this, not
    # against the nominal energy, so integral non-linearity and fit systematics of the baseline cancel.
    centroid_ch: float = 0.0
    centroid_err_ch: float = 0.0


@dataclass
class BackgroundBand:
    label: str
    lo: float
    hi: float
    counts: int


@dataclass
class Baseline:
    calibration: EnergyCalibration
    resolution: Resolution
    lines: list[BaselineLine]
    n_channels: int
    source_peaks: list[float]  # keV, features present in healthy source spectra
    background_peaks: list[float] = field(default_factory=list)
    background_bands: list[BackgroundBand] = field(default_factory=list)
    background_live_time: float | None = None
    tolerances: Tolerances = field(default_factory=Tolerances)
    fit: FitOptions = field(default_factory=FitOptions)
    profile: str = "hpge"
    created: str = ""

    # ---- construction -------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        source: Spectrum,
        background: Spectrum | None = None,
        reference_energies=None,
        tolerances: Tolerances | None = None,
        profile="hpge",
    ) -> Baseline:
        """Build from a known-good check-source spectrum (and optionally a background run).

        ``profile`` ("hpge", "nai" or a `DetectorProfile`) supplies defaults for the reference
        lines, peak model and tolerances; explicit arguments override it.
        """
        from .profiles import get_profile

        prof = get_profile(profile)
        if source.calibration is None:
            raise ValueError("the baseline source spectrum needs a nominal calibration")
        tol = tolerances or prof.tolerances
        energies = list(reference_energies if reference_energies is not None else prof.reference_energies)
        opts = prof.fit
        nominal = source.calibration
        kw = {"step": opts.step, "roi_fwhm": opts.roi_fwhm, "search_fwhm": opts.search_fwhm,
              "multiplet_fwhm": opts.multiplet_fwhm,
              "group_fwhm_keV": lambda e: float(prof.initial_resolution.fwhm(e))}

        first_pass = measure_lines(
            source.counts, nominal, energies, lambda e: float(prof.initial_resolution.fwhm(e)), **kw
        )
        fits = measure_lines(  # second pass seeded with the first-pass widths
            source.counts, nominal, energies,
            lambda e: _seed_width(e, energies, first_pass, nominal, prof.initial_resolution), **kw,
        )
        found = [(e, f) for e, f in zip(energies, fits, strict=True) if f is not None]
        if not found:
            raise ValueError("no reference line found in the baseline spectrum")

        e_found = np.array([e for e, _ in found])
        centroids = np.array([f.centroid for _, f in found])
        cent_err = np.array([f.centroid_err for _, f in found])
        if len(found) >= 2:
            cal = fit_calibration(centroids, e_found, cent_err, order=min(prof.calibration_order, len(found) - 1))
        else:  # a single line cannot calibrate: keep the nominal calibration
            cal = nominal

        fwhm = np.array([f.fwhm * float(cal.slope(f.centroid)) for _, f in found])
        fwhm_err = np.array([f.fwhm_err * float(cal.slope(f.centroid)) for _, f in found])
        if prof.resolution_kind == "power":
            fixed = None if len(found) >= 3 else prof.power_law_exponent
            res: Resolution = fit_power_law(e_found, fwhm, fwhm_err, fixed_p=fixed)
        elif len(found) >= 2:
            res = fit_resolution(e_found, fwhm, fwhm_err)
        else:
            res = ResolutionModel(0.0, float(fwhm[0] ** 2 / e_found[0]))

        lib = all_lines()
        lines = []
        for (e, f), w, we in zip(found, fwhm, fwhm_err, strict=True):
            near = [ln for ln in lib if abs(ln.energy - e) < 0.5]
            lines.append(BaselineLine(e, near[0].label if near else f"{e:.1f} keV", float(w), float(we),
                                      f.area / source.live_time, float(f.centroid), float(f.centroid_err)))

        base = cls(
            calibration=cal, resolution=res, lines=lines, n_channels=source.n_channels, source_peaks=[],
            tolerances=tol, fit=opts, profile=prof.name,
            created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        base.source_peaks = base._peak_energies(source, tol.baseline_peak_threshold)
        if background is not None:
            base.background_peaks = base._peak_energies(background, tol.baseline_peak_threshold)
            base.background_live_time = background.live_time
            base.background_bands = [
                BackgroundBand(label, lo, hi, background.band_counts(lo, hi, cal))
                for label, lo, hi in base._band_definitions()
            ]
        return base

    def _band_definitions(self) -> list[tuple[str, float, float]]:
        bands = [("20-3000 keV total", 20.0, 3000.0)]
        bands += [(f"{lo:g}-{hi:g} keV", float(lo), float(hi)) for lo, hi in BACKGROUND_BANDS]
        last_hi = -np.inf
        for e in BACKGROUND_WINDOW_LINES:  # skip windows that would overlap the previous one (scintillators)
            half = 1.5 * float(self.resolution.fwhm(e))
            if e - half >= last_hi:
                bands.append((f"{e:.1f} keV line", e - half, e + half))
                last_hi = e + half
        return bands

    @property
    def finder_band(self) -> int:
        """Block size (channels) over which the peak search keeps one filter width."""
        return int(max(32, min(256, self.n_channels // 8)))

    def fwhm_ch(self, calibration: EnergyCalibration):
        """Expected peak FWHM in channels as a function of channel number."""
        return lambda ch: float(self.resolution.fwhm(calibration(ch)) / calibration.slope(ch))

    def _peak_energies(self, spectrum: Spectrum, threshold: float) -> list[float]:
        cal = self.calibration
        min_ch = int(np.ceil(float(cal.channel(self.tolerances.min_search_keV))))
        cands = find_peak_candidates(spectrum.counts, self.fwhm_ch(cal), threshold, min_channel=min_ch,
                                     band=self.finder_band)
        return sorted(float(cal(c.channel)) for c in cands)

    # ---- persistence ----------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "schema": 2,
            "created": self.created,
            "profile": self.profile,
            "calibration": self.calibration.to_dict(),
            "resolution": self.resolution.to_dict(),
            "lines": [asdict(ln) for ln in self.lines],
            "n_channels": self.n_channels,
            "source_peaks": self.source_peaks,
            "background_peaks": self.background_peaks,
            "background_bands": [asdict(b) for b in self.background_bands],
            "background_live_time": self.background_live_time,
            "tolerances": asdict(self.tolerances),
            "fit": asdict(self.fit),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Baseline:
        return cls(
            calibration=EnergyCalibration.from_dict(d["calibration"]),
            resolution=resolution_from_dict(d["resolution"]),
            lines=[BaselineLine(**ln) for ln in d["lines"]],
            n_channels=d["n_channels"],
            source_peaks=list(d["source_peaks"]),
            background_peaks=list(d.get("background_peaks", [])),
            background_bands=[BackgroundBand(**b) for b in d.get("background_bands", [])],
            background_live_time=d.get("background_live_time"),
            tolerances=Tolerances(**d.get("tolerances", {})),
            fit=FitOptions(**d.get("fit", {})),
            profile=d.get("profile", "hpge"),
            created=d.get("created", ""),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Baseline:
        return cls.from_dict(json.loads(Path(path).read_text()))


def _seed_width(energy, energies, fits, nominal, default: Resolution) -> float:
    """First-pass FWHM (keV) of the line at ``energy`` if it was found, else the profile default."""
    for e, f in zip(energies, fits, strict=True):
        if e == energy and f is not None:
            return f.fwhm * float(nominal.slope(f.centroid))
    return float(default.fwhm(energy))
