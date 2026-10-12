# GammaGuard

**Gamma spectroscopy anomaly detection and detector QA, for HPGe and NaI(Tl).**
Detect energy-calibration drift, loss of resolution, background changes and suspicious spectral features in simulated or real
detector measurements.

`Python` · `Signal processing` · `Poisson statistics` · `Peak fitting` · `QA`

> **Status: v0.2.** Tested on a simulator with injected faults *and* sanity-checked on two open real datasets (one NaI(Tl), one
> HPGe). The real-data results are mixed, and the repo says so: some of the published goodness-of-fit figures were **not**
> reproduced, and the real data exposed three bugs that the HPGe simulator never showed (plus two more found while building the NaI
> simulator; all fixed, all with regression tests). Read
> [docs/validation.md](docs/validation.md) before trusting it with a real detector, and see [Limitations](#limitations).

## What it checks

GammaGuard compares a new measurement with a **baseline** built from known-good data and grades the change `PASS` / `WARN` / `FAIL`.

| Failure mode | Check | How |
|---|---|---|
| Energy-calibration drift | `calibration` | Poisson-likelihood fits of reference lines (blended lines fitted together); shift of each line **relative to the baseline**, split into **offset (keV)** and **gain (%)** |
| Loss of resolution | `resolution` | Measured vs baseline FWHM per line, plus an inverse-variance mean |
| Background change | `background` | Exact conditional-binomial rate tests on broad bands and ambient-line windows (Pb-214, Bi-214, K-40, Tl-208, 511), Bonferroni-corrected |
| Suspicious spectral features | `unexpected_peaks` | Poisson-aware peak search; peaks absent from the baseline are reported with nuclide hints |
| ADC / digitiser faults | `channel_anomalies` | Hot and dead channels via an exact neighbour test, with peeling and a strength-aware peak mask |
| Acquisition problems | `acquisition` | Dead time, channel-count mismatch |

An alarm needs the **whole confidence interval** beyond the tolerance (defaults: 0.10 × FWHM line shift to warn, 8 % width, 25 %
background), so low statistics can't raise false alarms. A check that can't reach a verdict says so (`WARN` for coarse precision,
`FAIL` for missing reference peaks, `SKIPPED` without a reference).

## Detector support

| | `hpge` | `nai` |
|---|---|---|
| Peak model | Gaussian + linear background | Gaussian + linear + Compton step; blended pairs fitted together (Co-60 1173/1332) |
| Resolution law | `FWHM² = a + bE` | `FWHM = k·E^p` |
| Default reference lines | 303, 356, 662, 1173, 1332 keV (Ba-133 + Cs-137 + Co-60) | 59.5, 662, 1173, 1332 keV (Am-241 + Cs-137 + Co-60) |
| Single-line baseline (daily Cs-137 check) | yes | yes |

Pick one with `profile="hpge" | "nai"` (or `--profile` / `--detector` on the CLI), or pass your own reference energies and tolerances.

## Quick start

```bash
pip install -e ".[dev]"      # numpy, scipy (+ pytest, ruff, matplotlib)
gammaguard demo              # end-to-end tour on simulated data
gammaguard demo --detector nai
```

A realistic workflow from the command line (HPGe):

```bash
# 1. capture a baseline while the detector is known to be healthy
gammaguard simulate --kind source     --live-time 1800  --seed 1 --out good_source.csv
gammaguard simulate --kind background --live-time 14400 --seed 2 --out good_bkg.csv
gammaguard baseline --source good_source.csv --background good_bkg.csv --out baseline.json

# 2. routine check, e.g. from cron; exit code 0 = pass, 1 = warn, 2 = fail
gammaguard simulate --seed 4 --fault gain_drift:0.0005 --fault hot_channel:2200:0.5 --out today.csv
gammaguard check --baseline baseline.json today.csv
```

```text
GammaGuard QA report (source mode): FAIL
  [PASS   ] acquisition        154754 counts in 600 s live time
  [FAIL   ] calibration        largest shift vs baseline +0.670 keV at Co-60 1332.5 keV (warn at 0.20); fit: offset +0.007 keV, gain +0.0495 %
  [PASS   ] resolution         mean FWHM ratio 1.006 +- 0.005 vs baseline; largest deviation 1.018 at Cs-137 661.7 keV (1.51 vs 1.48 keV)
  [PASS   ] unexpected_peaks   no peaks beyond the 13 baseline features (threshold 5.5 sigma)
  [FAIL   ] channel_anomalies  1 anomalous channel(s): ch 2200 hot (320 vs 9.0)
```

The same for a scintillator (`--detector nai` when simulating, `--profile nai` for the baseline), here with a +1 % gain drift and 20 % worse resolution:

```bash
gammaguard simulate --detector nai --kind source     --live-time 1800  --seed 1 --out nai_good.csv
gammaguard simulate --detector nai --kind background --live-time 14400 --seed 2 --out nai_bkg.csv
gammaguard baseline --profile nai --source nai_good.csv --background nai_bkg.csv --out nai_base.json
gammaguard simulate --detector nai --seed 4 --fault gain_drift:0.01 --fault resolution_loss:1.2 --out nai_today.csv
gammaguard check --baseline nai_base.json nai_today.csv
```

```text
GammaGuard QA report (source mode): FAIL
  [PASS   ] acquisition        49046 counts in 600 s live time
  [WARN   ] calibration        largest shift vs baseline +16.044 keV at Co-60 1332.5 keV (warn at 6.50); fit: offset -0.069 keV, gain +1.1395 %
  [FAIL   ] resolution         mean FWHM ratio 1.206 +- 0.011 vs baseline; largest deviation 1.213 at Am-241 59.5 keV (14.82 vs 12.21 keV)
  [PASS   ] unexpected_peaks   no peaks beyond the 9 baseline features (threshold 5.5 sigma)
  [PASS   ] channel_anomalies  no hot or dead channels among 674 tested
```

(`--format md|json` and `--output FILE` produce machine-readable reports.) Real spectra go in as CSV (`# live_time: …`,
`# calibration: c0,c1`, then `channel,counts`); see [`io.py`](src/gammaguard/io.py).

```python
from gammaguard import Baseline, run_qa
from gammaguard.io import read_spectrum

baseline = Baseline.build(read_spectrum("good_source.csv"), read_spectrum("good_bkg.csv"), profile="hpge")
report = run_qa(baseline, read_spectrum("today.csv"))          # mode="background" for source-free runs
print(report.to_text()); print(report.check("calibration").metrics["gain_change_pct"])
```

## Real-data validation

Two open datasets, **downloaded on demand and never bundled** (the ETH set is non-commercial-use; see [docs/data.md](docs/data.md)):

```bash
gammaguard fetch-data all
python validation/validate_nai_rll.py && python validation/validate_nai_qa.py && python validation/validate_hpge_crm.py
pytest -m realdata
```

| Dataset | What was tested | Result |
|---|---|---|
| NaI(Tl): Breitenmoser et al. 2022 (4 crystals, 10 sources, 10 backgrounds) | The paper's claim of a linear, offset-free calibration and a power-law resolution | **Partly confirmed.** Linear and offset-free to ~0.3 % of range; FWHM 7.3-8.1 % @ 662 keV. The quoted goodness-of-fit (R² > 0.9999 / > 0.997) is reproduced for only 2 of 4 crystals (calibration) and for none (resolution: 0.988-0.992) |
| NaI(Tl): same data | Crystals 1, 2, 4 graded against crystal 3 (real, independently measurable differences) | Widths agree with independent 9-line fits to **1 %** (0.960/0.961, 0.869/0.875, 0.882/0.892); line shifts to ≈ 4 keV (≈ 0.08 FWHM); crystal 3 vs itself PASS |
| NaI(Tl): same data | 9 background measurements vs the first | 9/9 PASS, total rates within 0.4 % |
| HPGe: Huynh & Truong (ORTEC GMX35P4-70, 176 spectra, 2013-2022) | 270 natural lines in 78 spectra | 270/270 found; reading error sd 0.14 keV with a systematic energy pattern (−0.18 … +0.16 keV, integral non-linearity); FWHM 1.88 keV @ 1332 keV |
| HPGe: same data | Same-material spectra vs a high-statistics baseline | 38 of 43 PASS (5 WARN: line shifts up to 0.2 keV) |
| HPGe: same data | Th-ore and K-salt spectra vs a U-ore baseline (known to be different) | all flagged; 62 % of the named peak hints are Th-chain nuclides |
| HPGe: same data | Backgrounds vs the longest one | Jan/Oct-2016 PASS (×0.99, ×1.01); PMMA-container background FAIL (×1.66) |

## Simulated benchmark

100 trials per case, fresh baseline and spectrum each trial (full tables in [docs/benchmark.md](docs/benchmark.md)):

| Case | Check | HPGe caught | NaI caught |
|---|---|---|---|
| Healthy source run / background run | – | **0 % / 0 %** false alarms | **0 % / 0 %** false alarms |
| Gain drift below tolerance (HPGe +0.01 %, NaI +0.2 %) | calibration | 0 % | 0 % |
| Gain drift above (HPGe +0.03 %, NaI +0.8 %) | calibration | 100 % | 97 % |
| Resolution ×1.04 / ×1.10 (HPGe), ×1.04 / ×1.12 (NaI) / ×1.25-1.30 | resolution | 0 % / 71 % / 100 % | 0 % / 89 % / 100 % |
| Background ×1.10 / ×1.50 / ×2.50 | background | 0 % / 100 % / 100 % | 0 % / 100 % / 100 % |
| Unexpected line, weak / strong (HPGe 120 / 600 ct, NaI 180 / 900 ct) | unexpected_peaks | 25 % / 98 % | 85 % / 98 % |
| Hot channel, small / large (HPGe +30 / +300 ct, NaI +180 / +1200 ct) | channel_anomalies | 52 % / 100 % | 99 % / 100 % |
| Dead channel (HPGe ~37 ct/ch, NaI ~100 ct/ch) | channel_anomalies | 99 % | 99 % |

Detection of weak or sparse features is limited by counting statistics; the full tables show where (for instance a dead HPGe channel
at ~10 counts/channel is never found). 1,600 further healthy trials (`validation/null_false_alarms.py`) produced no WARN or FAIL.
These are simulation results, not field performance.

## Trend monitoring

A single-run tolerance only fires once a drift has already crossed the limit.
[`examples/slow_drift_trend.py`](examples/slow_drift_trend.py) ramps the gain from day 40 and grades 120 daily checks:

| Method | First alarm | Shift of the 1332.5 keV line vs baseline |
|---|---|---|
| Single-run tolerance (WARN) | day 81 | +0.272 keV |
| Shewhart 3σ on the daily series | day 49 | +0.065 keV |
| CUSUM / EWMA | day 47 | +0.049 keV |

Trend charts fire about five weeks before the single-run WARN here. At this noise level CUSUM/EWMA gain only ~2 days over a plain 3σ chart.

## Repository layout

```
src/gammaguard/
  spectrum.py  calibration.py  resolution.py  isotopes.py  io.py     data model (two resolution laws)
  stats/       poisson.py (exact tests, deviance, Currie)  control.py (Shewhart / CUSUM / EWMA)
  peaks/       finder.py (Poisson-aware peak search)       fit.py (Poisson-likelihood multiplet fit)
  simulate/    generator.py (HPGe + NaI forward model)     faults.py (ten fault injectors)
  qa/          profiles.py  baseline.py  measure.py  checks.py  report.py   run_qa()
  datasets/    registry.py  fetch.py  eth_rll.py  hpge_crm.py   open datasets (downloaded, not bundled)
  bench.py     cli.py
tests/         87 tests: statistics, fitting, multiplets, simulator, every check, both detectors, CLI, dataset layer, real-data guards
validation/    validate_nai_rll.py  validate_nai_qa.py  validate_hpge_crm.py  null_false_alarms.py  results/*.json
examples/      slow_drift_trend.py
docs/          design.md (statistics and rationale)  validation.md  benchmark.md  data.md
```

## Limitations

* **Validation is by consistency, not by a field trial.** The real datasets contain no injected faults, so detection rates come from the
  simulator. NaI statistics rest on an *estimated* effective counting time (the data are linearized count rates); the HPGe set is
  rebinned. Hot/dead-channel statistics have not been validated on raw MCA channel data.
* The paper's NaI goodness-of-fit was not reproduced (gross-spectrum Gaussian + step fits carry a few keV of model systematics).
* Peak model: Gaussian (+ linear + optional step). No low-energy tails, escape/sum peaks or X-ray structure; NaI lines below ~300 keV were
  not reliable in gross spectra.
* The unexpected-peak check assumes the baseline has at least the statistics of routine measurements.
* Channels near peaks are not tested for hot/dead behaviour; a dead channel needs ~20+ expected counts to be seen; a spurious peak
  within the match tolerance of a baseline feature is masked.
* No efficiency / peak-area stability check yet.
* Nuclide data in [`isotopes.py`](src/gammaguard/isotopes.py) are illustrative; verify before use beyond simulation.
* Tolerances are defaults, not recommendations for any facility.
* GitHub Actions runs lint, the offline test suite and a CLI smoke test on Python 3.10-3.13 (all passing at the first run). The
  real-data tests are not part of CI: they need the downloaded datasets.

## Roadmap

1. **More real data**: raw-channel MCA files (IAEA SPE, ANSI N42.42, vendor formats), spectra with *known* injected faults or recorded drift
   events, to measure detection rather than consistency.
2. **Peak model**: low-energy tails, escape/sum peaks, nonproportionality for NaI; reliable low-energy lines on gross spectra.
3. **Efficiency check**: peak-area rates with decay correction.
4. **Unexpected peaks**: compare against baseline *counts* (rate test) instead of a feature list, to remove the statistics assumption; Poisson ROI
   significance for sparse spectra.
5. **History**: persistent run store and a `gammaguard trend` command using the control charts.
6. **Reports**: plots / HTML report; baseline refresh policy.

## Development

```bash
pip install -e ".[dev]"
pytest                          # ~5 s offline; the 4 real-data tests are skipped unless the datasets are cached
pytest -m realdata              # after: gammaguard fetch-data all
ruff check src tests examples validation
gammaguard benchmark --detector nai --trials 100
```

## License

Copyright 2026 Sobhan Shahbazi. The code is licensed under the **Apache License 2.0** ([LICENSE](LICENSE), [NOTICE](NOTICE)).
The validation datasets are third-party and **not** covered by it; see [docs/data.md](docs/data.md) for their licences and citations.
