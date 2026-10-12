# Validation on real spectra

GammaGuard was developed against a simulator, which can only confirm that the code does what its author intended. This
page records what happened when it met two **real, open-access** datasets: one NaI(Tl), one HPGe. Everything is
reproducible with the scripts in [`validation/`](../validation) once the data are fetched
(`gammaguard fetch-data all`; the datasets keep their own licences and are not bundled: see [data.md](data.md)).

| Dataset | Detector | Used for |
|---|---|---|
| Breitenmoser et al. 2022 (ADGEO 57, 89) + ETH Research Collection data | NaI(Tl), 4 crystals × 10.2 × 10.2 × 40.6 cm, 1024 ch, 10 sources + 10 backgrounds | peak fitting vs the paper's claims; detector-to-detector QA; background stability |
| Huynh & Truong, Zenodo 10.5281/zenodo.23093533 | HPGe, ORTEC GMX35P4-70, 176 spectra 2013–2022 (78 of pure RGU-1/RGTh-1/RGK-1 used) | natural-line calibration/resolution; cross-material unexpected-peak test with known truth; channel-check false alarms; backgrounds |

## What "confirmation" can and cannot mean here

* The ADGEO paper reports **model forms and goodness-of-fit** (`H = d1·E` with R² > 0.9999 and LOO error < 0.02 %;
  `FWHM(H) = b1·H^b2` with R² > 0.997 and LOO error < 0.2 %), but the fitted parameter *values* are only in a figure. So
  the test is an independent re-fit, not a comparison of numbers.
* The ETH data are **count rates after the spectrometer's onboard spectral linearization**, not raw counts. GammaGuard's
  Poisson statistics need counts, so each spectrum is scaled by an *effective counting time* estimated from its
  channel-to-channel scatter (of order 10⁴–10⁵ s here). It reproduces the noise level but is an estimate, and the
  interpolation makes neighbouring channels correlated. Statistical errors on NaI results are therefore approximate;
  the Birge-ratio error scaling in the peak fit absorbs part of the mismatch.
* The HPGe dataset was **rebinned to 0.5 keV with fractional counts** after the authors refitted every calibration. That
  makes it unsuitable for channel-level fault statistics and means calibration drift is already removed.
* Neither dataset contains injected faults. "Faults" below are real, independently measurable differences: another crystal,
  another material, another container.

## 1. NaI(Tl): the paper's calibration and resolution claims (`validate_nai_rll.py`)

Nine lines per crystal, fitted in gross spectra with GammaGuard's Gaussian + Compton-step model (blended pairs fitted together):
Unat 351.9, 609.3 · Cs-137 661.7 · Y-88 898.0, 1836.1 · Co-60 1173.2, 1332.5 · Knat 1460.8 · Thnat 2614.5 keV.
Lines below 300 keV are excluded: in gross spectra they sit on X-ray and scatter structure next to the threshold, and
the locator latches onto a strong 78 keV feature instead (Eu-152 122 keV and Cd-109 88 keV, which the paper fits on net
spectra, were not recoverable from gross spectra, and Eu-152 344 keV was not located reliably).

| Crystal | keV/ch | Free-fit offset (keV) | Cal. R² | Cal. LOO err. | Res. R² | Res. LOO err. | FWHM@662 | FWHM@1332 |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.017 | +8.7 ± 3.4 | 0.99963 | 0.039 % | 0.988 | 1.8 % | 7.91 % | 6.25 % |
| 2 | 3.012 | +10.1 ± 4.5 | 0.99967 | 0.039 % | 0.992 | 1.1 % | 7.28 % | 5.66 % |
| 3 | 2.991 | −4.9 ± 1.5 | 0.99998 | 0.002 % | 0.989 | 1.7 % | 8.11 % | 6.57 % |
| 4 | 2.997 | +6.1 ± 3.4 | 0.99990 | 0.016 % | 0.990 | 1.4 % | 7.34 % | 5.81 % |
| sum | 3.006 | +6.9 ± 3.0 | 0.99983 | 0.021 % | 0.990 | 1.5 % | 7.73 % | 6.10 % |
| *paper* | – | *none (H = d1·E)* | *> 0.9999* | *< 0.02 %* | *> 0.997* | *< 0.2 %* | – | – |

**Confirmed:** the response is linear and essentially offset-free (free-fit offsets of −5 to +10 keV, about 0.3 % of the range at most,
with inconsistent sign across crystals); resolution is 7.3–8.1 % at 662 keV and 5.7–6.6 % at 1332 keV, FWHM ∝ E^0.64–0.70
in channel units, as expected for a large airborne NaI(Tl) crystal; all nine lines were found on all four crystals.
**Not reproduced:** the paper's goodness-of-fit. Calibration R² reaches the quoted 0.9999 only for crystals 3 and 4 (and the
LOO error only for those), and the resolution R² is 0.988–0.992, below 0.997, with LOO errors of 1–2 % instead of 0.2 %.
Maximum residuals of the origin-forced calibration are 4.9 keV (crystal 3) up to 37.6 keV (crystal 1). The authors used net
spectra, a different baseline model and probably more careful line selection; our gross-spectrum Gaussian + step fits
carry model systematics of a few keV per line (the same Cs-137 peak moves by ~1.6 keV between step / no-step models).
That is small against the NaI tolerance (≈ 0.1 FWHM ≈ 5 keV) but visible in a 9-point R².

## 2. NaI(Tl): QA across detectors and backgrounds (`validate_nai_qa.py`)

**Experiment A: crystals as real faults.** A mixed check source is emulated by adding the Cs-137 and Co-60 rate spectra. The
baseline comes from crystal 3 (Cs-137 8.45 %, Co-60 6.48 % and 6.12 % FWHM); the others are graded against it. The line shifts
and widths GammaGuard reports (3 lines) are compared with those predicted from the independent smooth 9-line fits of §1.

| Crystal | Overall | Shift of 662 / 1173 / 1332 keV line: GammaGuard | independent | FWHM ratio: GammaGuard | independent |
|---|---|---|---|---|---|
| 3 (itself) | **PASS** | −0.0 / −0.2 / +0.2 keV | 0 / 0 / 0 | 0.999 | 1.000 |
| 1 | WARN (calibration) | −8.4 / −9.9 / −12.7 | −11.3 / −9.5 / −8.9 | 0.960 | 0.961 |
| 2 | WARN (calibration, resolution) | −7.7 / −6.3 / −8.8 | −11.3 / −8.4 / −7.4 | 0.869 | 0.875 |
| 4 | WARN (resolution, unexpected peak) | −3.2 / −2.1 / −4.5 | −6.5 / −3.0 / −1.9 | 0.882 | 0.892 |

* **Resolution agrees to within 1 %** (0.960 vs 0.961, 0.869 vs 0.875, 0.882 vs 0.892) and GammaGuard correctly reports that
  crystals 2 and 4 are ~12 % *narrower* than the baseline. Shifts agree to within ≈ 4 keV per line (≈ 0.08 FWHM), against
  shifts of 2–13 keV; that is the size of the model systematic noted above.
* The default tolerance (shift 0.10 × FWHM ≈ 4.6–6.5 keV) is exceeded by crystals 1 and 2, as it should be.

**Experiment B: ten background measurements** (one per source measurement), graded in background mode against the first: all
nine PASS, with total rates within 0.4 % of each other (×0.996 … ×1.002) and no band flagged.

## 3. HPGe: natural lines, cross-material tests, channel checks (`validate_hpge_crm.py`)

**H1: natural-line calibration and resolution (78 spectra of RGU-1, RGTh-1, RGK-1).** Lines are fitted within ±3 FWHM of their
nominal position (the authors' refit puts them there). 270 of 270 expected lines were found.
Reading error versus nominal energy: mean −0.005 keV, sd 0.138 keV, 95th percentile 0.20 keV, maximum 0.42 keV.

| Line (keV) | n | Reading error (keV) | FWHM (keV) |
|---|---|---|---|
| 238.63 | 20 | −0.183 ± 0.006 | 1.10 |
| 351.93 | 44 | −0.178 ± 0.034 | 1.21 |
| 583.19 | 20 | −0.003 ± 0.013 | 1.37 |
| 609.31 | 44 | −0.024 ± 0.026 | 1.38 |
| 911.20 | 20 | +0.112 ± 0.028 | 1.59 |
| 1120.29 | 44 | +0.118 ± 0.012 | 1.73 |
| 1460.82 | 14 | +0.159 ± 0.183 | 1.95 |
| 1764.49 | 44 | +0.110 ± 0.031 | 2.10 |
| 2614.51 | 20 | −0.157 ± 0.092 | 2.54 |

* The errors are **systematic with energy** (−0.18 keV at 240–350 keV, +0.11 to +0.16 keV at 0.9–1.8 MeV), far above the
  statistical errors. That is the integral non-linearity the authors' *linear* refit leaves behind; it is a property of the
  detector/electronics and exactly the kind of thing that should not count as drift (see §4, first item).
* FWHM interpolates to **1.88 keV at 1332.5 keV** (the manufacturer's specification is not part of the dataset).
  The median FWHM at 609 keV is 1.37-1.39 keV in 2015-2016 (n = 19 each) and the few single spectra of 2013-2022 lie between 1.38 and
  1.49 keV: too few spectra outside 2015-2016 to call a long-term trend.

**H2: unexpected-peak check with ground truth.** Baseline = the highest-statistics RGU-1 spectrum (S207, 52 h, 220 M counts).
Because the check compares with a feature list built at the baseline's statistics, the same-material tests are restricted to
spectra with no more counts than the baseline.

| Test set | n | PASS / WARN / FAIL | Unexpected peaks | Remarks |
|---|---|---|---|---|
| RGU-1 (same material) | 43 | 38 / 5 / 0 | 2 in 1 spectrum | 5 WARNs are calibration: line shifts up to 0.2 keV |
| RGTh-1 (thorium ore) | 20 | 0 / 1 / 19 | 585 in 20 spectra | the U-chain reference lines are absent → calibration FAIL; 61 peaks have a library hint, **62 % of them Th-chain nuclides** (Tl-208, Ac-228, Pb-212, Bi-212) |
| RGK-1 (potassium salt) | 14 | 0 / 1 / 13 | 8 in 7 spectra | reference lines absent → calibration FAIL |

**H3: hot/dead-channel check on real, rebinned spectra.** 0 flags in the 43 RGU-1 and 14 RGK-1 spectra; 12 flags in 5 of the 20 RGTh-1
spectra (Th-ore spectra have the strongest lines). The rebinned counts are not Poisson-independent, so this is a *false-alarm*
stress test, not a sensitivity test.

**H4: backgrounds**, reference = the longest (144 h, Feb 2016): the Jan-2016 box background (×0.99) and Oct-2016 lead-shield
background (×1.01) PASS; the April-2016 **PMMA (incense) container background is FAIL (×1.66** in total rate; 20-100 and
100-300 keV bands flagged), a genuinely different measurement setup.

## 4. What the real data taught us (bugs the HPGe simulator could not show)

Items 1-3 and the matched-filter half of item 4 surfaced on **real spectra**; the despiking half of item 4 and item 5 were found
while building the **NaI simulator**. None of them was visible in the original 1,600 simulated healthy HPGe trials, and each has a
regression test.

1. **Drift was measured against the wrong thing.** The calibration check compared line positions with the *nominal energies*
   through the baseline's linear fit, so the baseline's own non-linearity residuals (±0.15 keV, H1) were counted as drift on every
   run: **0 of 43** healthy same-detector HPGe spectra passed the default tolerance (42 of 43 only at 0.25 × FWHM). Drift is now
   measured against where the *baseline* measured each line: 38 of 43 pass.
2. **A degenerate peak fit relocated features.** For continuum edges the Gaussian fit collapses (area ≈ 0 ± 10⁵) and its centroid
   wanders up to 2 FWHM away, so the feature no longer matched the baseline list and a baseline compared with its own data raised
   "unexpected peaks" (6 of 9 NaI background pairs). Fits that have not converged on a peak no longer move the candidate.
3. **The hot/dead-channel mask was too narrow for strong lines.** At 10⁵-10⁶ counts the curvature bias of a 5-channel neighbour mean
   exceeds the Poisson noise far beyond ±2 FWHM: 3,213 flags in the 20 Th-ore spectra. The mask now grows with each fitted peak's
   strength: 12 flags remain.
4. **Peak location on scintillators.** A raw smoothed-count scan latches onto the rising continuum; the locator now uses the zero-sum
   matched filter, after a 3-point median that removes single-channel spikes (a strong spike in the search window used to drag the
   calibration check into a false alarm).
5. **Multiplet grouping flipped at the threshold.** The Co-60 pair (159 keV apart) sits at 2.5 FWHM in NaI; grouping used the
   measured widths, so the second pass fitted the lines separately and produced a 25 % width error. Grouping now uses a smooth resolution
   law and a 3-FWHM threshold.

## 5. Limits of this validation

* Real-data checks are **sanity and consistency tests**, not a field-performance claim. No dataset here contains injected faults at
  known magnitudes, so detection rates come only from the simulator ([benchmark.md](benchmark.md)).
* The NaI statistics rest on an estimated effective counting time (see above); the HPGe set is rebinned. Channel-level Poisson
  tests are not validated on real raw-channel data at all.
* The unexpected-peak check assumes the baseline has at least the statistics of the routine measurement (H2); with a poorer baseline
  it reports weak but real lines (U-235-chain lines at 286, 324, 330 and 388 keV in RGU-1 appeared that way).
* One detector of each type, one laboratory each. Tolerances (0.10 × FWHM shift, 8 % width, 25 % background) are defaults, not
  recommendations for any particular facility.

Reproduce:

```bash
gammaguard fetch-data all
python validation/validate_nai_rll.py     # §1  → validation/results/nai_rll.json
python validation/validate_nai_qa.py      # §2  → validation/results/nai_qa.json
python validation/validate_hpge_crm.py    # §3  → validation/results/hpge_crm.json
pytest -m realdata                        # fast regression guards on the same data
```
