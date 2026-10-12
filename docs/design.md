# GammaGuard design notes

This document explains *why* the code is shaped the way it is: the model of a healthy detector, the statistics behind each
check, and the limits of each. For usage see the [README](../README.md); for evidence see [benchmark.md](benchmark.md)
(simulated) and [validation.md](validation.md) (real spectra).

## 1. Core idea: baseline-relative QA

GammaGuard does not try to decide from a single spectrum whether a detector is "good". It compares a measurement with a
**baseline** captured from known-good data, and grades the *change*:

```
 healthy check-source run ─┐
 healthy background run  ──┼─► Baseline (JSON) ─┐
 detector profile ─────────┘                    ├─► run_qa() ─► QAReport (PASS / WARN / FAIL per check)
 new spectrum ──────────────────────────────────┘
```

Reasons:

* Tolerances that matter operationally ("a line may not move by more than a fraction of its FWHM") are relative to the
  detector's own behaviour, not to textbook values.
* Features that are normal for *this* detector (Compton edges, weak lines, ambient lines, escape peaks, integral
  non-linearity) are recorded in the baseline instead of being modelled from first principles.

The real-data validation made this principle stricter: **drift is measured against where the baseline measured each line
(its fitted centroid), not against the nominal line energy.** The baseline's own residuals against nominal energies (about
±0.15 keV on the HPGe set from integral non-linearity and fit systematics) are properties of the baseline, and counting them as
drift made every healthy spectrum look drifted (validation.md §4).

## 2. Module map

| Layer | Modules | Responsibility |
|---|---|---|
| Data model | `spectrum`, `calibration`, `resolution`, `isotopes`, `io` | `Spectrum`, channel→energy polynomial, two FWHM(E) laws, line library, CSV/JSON files |
| Statistics | `stats.poisson`, `stats.control` | exact Poisson/binomial tests, deviance residuals, Currie limits, Shewhart/CUSUM/EWMA |
| Signal processing | `peaks.finder`, `peaks.fit` | Poisson-aware peak search, Poisson-likelihood multiplet fit |
| Simulation | `simulate.generator`, `simulate.faults` | forward model for HPGe and NaI(Tl) + ten fault injectors with ground-truth labels |
| QA | `qa.profiles`, `qa.baseline`, `qa.measure`, `qa.checks`, `qa.report`, `qa` (`run_qa`) | detector profiles, baseline, six checks, report |
| Data | `datasets.*` | registry, verified downloader, loaders for the open validation datasets (not bundled) |
| Tooling | `cli`, `bench` | CLI with CI-friendly exit codes, Monte Carlo benchmark |

Dependencies point downwards only: `qa` uses `peaks`, `stats`, `calibration`; nothing below `qa` knows about QA. File formats
never leak past `io` and `datasets`.

## 3. The physical model of a "healthy detector"

* **Calibration**: `E(ch) = c0 + c1·ch (+ c2·ch² …)`, fitted from peak centroids by weighted least squares.
* **Resolution**, one of two laws chosen by the detector profile:
  `FWHM² = a + b·E (+ c·E²)` (germanium: electronic noise + Fano statistics) or `FWHM = k·E^p` (scintillators, p ≈ 0.5-0.6).
* **Peak shape**: Gaussian integrated exactly over each channel (`ndtr` differences) on a linear background; for scintillators
  also a **Compton step** `S·Φ((μ−x)/σ)` under each peak (continuum higher on the low-energy side).
* **Multiplets**: lines closer than `multiplet_fwhm` FWHM are fitted *together*, with one shared width scale `s` (`σ_i = s·σ0_i`).
  A lone peak is just the N = 1 case. This is what makes the Co-60 pair (159 keV apart, ≈ 2.5 FWHM in NaI) well-posed.

The nominal calibration travels *with the spectrum* (`Spectrum.calibration`): it is what the acquisition software believes. The
simulator keeps the physical truth separately (`DetectorConfig.actual_calibration`), which is how a "drifted" detector is
represented: peaks land where the truth says but are labelled with the nominal calibration.

### Detector profiles

A `DetectorProfile` holds everything that differs between detector types and is copied into the baseline, so a saved baseline is
self-contained.

| | `hpge` | `nai` |
|---|---|---|
| Reference lines | 303, 356, 662, 1173, 1332 keV | 59.5, 662, 1173, 1332 keV |
| Resolution law | quadrature | power law |
| Compton step in peak fit | no | yes |
| Fit region (± FWHM) | 4 | 2 (wider regions drag in a continuum that is far from linear) |
| Peak-locating window (± FWHM) | 10 | 2.5 |
| Blend threshold (FWHM) | 2.5 | 3.0 |
| Peak-match tolerance for spectral features (FWHM) | 2 | 1 |
| Minimum search energy | 30 keV | 50 keV |

A baseline can be built from a single line (a daily Cs-137 check): the nominal calibration is kept and only the width scale is
fitted (exponent fixed by the profile).

## 4. Statistics, check by check

### 4.1 Peak fit (`peaks.fit`)

Fit in **channel space** (so a wrong calibration cannot bias the result). Minimises the Poisson deviance with
`scipy.optimize.least_squares` on signed deviance residuals (adaptive `x_scale="jac"`), an exact Poisson MLE, not a χ² with a
model-dependent weight. Parameter covariance comes from `(JᵀJ)⁻¹`, which equals the Fisher information for deviance residuals.
Uncertainties are inflated by `sqrt(deviance/dof)` when the fit is worse than Poisson noise allows (Birge ratio), so that
real-data model mismatch widens the errors instead of producing false certainty.

Hard-won details:

* The background level must start strictly positive. In a ~0.1 counts/channel region a zero start makes the likelihood singular
  and the optimiser "converges" with zero reported error.
* Fixed per-parameter scales stall on very high-statistics scintillator peaks (500 iterations without moving the width);
  Jacobian-based scaling converges in a few dozen.
* On NaI the Gaussian + step model is an approximation: simulated data show a 3-6 % low bias on the tied widths of a blended pair,
  and real spectra carry a ~1.6 keV model dependence at 662 keV. Both are the same in baseline and test and cancel in the comparison.

### 4.2 Peak search (`peaks.finder`)

A zero-sum, Gaussian-matched second-derivative filter (Mariscotti-style) whose width follows the expected local FWHM. The output
variance is propagated from the counts (`Σ k²·n`), floored so that a lone count is never a peak. Zero-sum ⇒ constant backgrounds
cancel; symmetric ⇒ linear backgrounds cancel.

*Limitation*: in nearly background-free regions the filter's `~√N` significance is pessimistic compared with an exact Poisson
tail test (a 15-count peak on 0.1 counts/ch is far more significant than 3.9σ).

### 4.3 Locating reference lines (`qa.measure`)

The expected position comes from the baseline calibration; a drifted detector needs a search. The group of lines is positioned by
scanning one common channel shift and scoring it with the **matched-filter significance** of a 3-point-median-filtered spectrum.
Two real-data lessons: scoring raw smoothed counts latches onto the rising continuum of a scintillator, and an un-despiked
single-channel spike inside the window can win the scan. Grouping of blended lines uses a **smooth resolution law**, not per-line
seed widths, otherwise a pair near the blend threshold is grouped differently on different passes.

### 4.4 Calibration check

For every reference line: `shift = slope · (centroid_now − centroid_baseline)` in keV, with the baseline centroid's error included.
A weighted linear fit `shift = z + g·E` separates **offset** (keV) from **gain** (fraction). The measured `(z, g)` produce a
*corrected calibration* that the later checks use, so a small drift does not masquerade as a spurious peak or a changed background
band. Tolerances are fractions of the baseline FWHM at the line (default 0.10 warn / 0.25 fail).

### 4.5 Resolution check

`ratio = FWHM_measured / FWHM_baseline` per line, with the baseline uncertainty propagated, plus an inverse-variance mean.
Narrowing beyond tolerance also warns (stale baseline or a bad fit; on the NaI crystals it flags a real 12 % difference).

### 4.6 Background check

For each band (six broad bands, the total, and narrow windows on the usual ambient lines, skipping windows that would overlap, which
matters for scintillators) the new count `n₁` in live time `t₁` is compared with the baseline `n₂` in `t₂` using the **exact
conditional binomial test**: under equal rates `n₁ | n₁+n₂ ~ Binomial(n₁+n₂, t₁/(t₁+t₂))`. It handles unequal live times and zero counts
without any Gaussian approximation. Family-wise error is controlled by Bonferroni over the bands.

### 4.7 Decision rule (all quantitative checks)

An alarm needs the **whole confidence interval** (± `min_significance` σ, or the Bonferroni-adjusted exact interval for rates) to lie
beyond the tolerance. "Significantly different from baseline" is not enough: with high statistics a 10 % background increase is hugely
significant against ratio = 1, and a +3σ fluctuation can carry the point estimate across a 25 % tolerance. The first implementation
used "point estimate beyond tolerance AND p < α" and raised 3 % false WARNs on sub-tolerance changes; the interval rule removed them.

Consequence: detection probability rises smoothly *through* the tolerance, not as a step at it (visible in the benchmark).

A check that cannot decide says so: centroid errors above `precision_warn_frac` of the tolerance → WARN ("precision too coarse");
missing reference lines → FAIL; no background reference → SKIPPED.

### 4.8 Spectral-feature checks

* **Unexpected peaks**: candidates from the peak finder at 5.5σ; a candidate is *expected* if it lies within `peak_match_fwhm` FWHM of a
  baseline feature. The baseline feature list is built at a **lower** threshold (3σ) than the test threshold (5.5σ) on purpose: a marginal
  real feature must not flip between "known" and "new" from run to run. Candidates much narrower than the expected FWHM are
  single-channel spikes and are left to the channel check. A candidate is only *relocated* by its peak fit if the fit converged
  (area ≥ 3σ, centroid within 0.75 FWHM); a degenerate fit on a Compton edge would otherwise wander off and no longer match.
  The check assumes the baseline has at least the statistics of the routine measurement; otherwise weak real lines are reported as new.
* **Hot / dead channels**: each channel is tested against its four nearest neighbours with the exact conditional test
  `n | n+S ~ Binomial(n+S, 1/5)` (`S` = neighbour sum). That is valid at any count level and has no nuisance parameter, unlike a
  Gaussian z-score or a plug-in Poisson tail (a first version used the latter and produced a false FAIL when the predictor
  fluctuated low). `S` is widened by `3·channel_rel_floor` to allow for curvature and differential non-linearity. Bonferroni over tested
  channels. A bad channel corrupts its neighbours' predictions, so channels are **peeled**: only the most significant one within ±2 is
  accepted per pass, replaced by the neighbour mean, and the test repeated.
  Where the neighbour mean is not a valid model, channels are **masked**: ±(2·FWHM+2) channels for narrow peaks (< 12 channels FWHM),
  ±(0.75·FWHM+2) for well-sampled ones, and, for every detected peak, wherever the Gaussian's curvature bias `1.25·h·|z²−1|/σ²` exceeds
  0.3 Poisson standard deviations. The last rule matters for strong lines: at 10⁵-10⁶ counts the bias exceeds the noise far from the peak.

## 5. Trend monitoring (`stats.control`)

Single-run tolerances only fire once a drift has already crossed the limit. Feeding a per-run metric (e.g. the line shift at
1332.5 keV) into Shewhart / CUSUM / EWMA charts, standardised on an in-control reference period, flags a sustained shift much earlier.
`examples/slow_drift_trend.py` demonstrates this.

## 6. Known limitations (v0.2)

* Real-data validation is a sanity/consistency check on two open datasets (one NaI(Tl), one HPGe), neither of which contains injected
  faults; detection rates come from the simulator. See [validation.md](validation.md) §5.
* The NaI dataset provides linearized count rates, so Poisson errors rest on an estimated effective counting time; the HPGe dataset is
  rebinned. **Channel-level (hot/dead) statistics have not been validated on raw MCA data.**
* Peak model is Gaussian (+ linear + optional step): no low-energy tails, no escape/sum peaks, no X-ray structure. Gross-spectrum
  fits of NaI lines below ~300 keV were not reliable on real data.
* Channels near peaks are not tested for hot/dead behaviour; a dead channel is detectable only where roughly 20+ counts per channel are
  expected; a spurious peak within `peak_match_fwhm` FWHM of a baseline feature is masked (a distorted known peak does still show in the
  resolution check).
* No efficiency / peak-area stability check yet (needs decay correction and source geometry).
* Nuclide data in `isotopes.py` are illustrative; verify before using them for anything but simulation and labelling.
* Calibration check in background mode relies on a calibration passed in from a recent source run.
