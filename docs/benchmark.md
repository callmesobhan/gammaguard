# Benchmark (simulated detectors)

Monte Carlo detection and false-alarm rates for both detector profiles. Every trial draws a **fresh baseline**
(a 1800 s check-source run and a 4 h background run) and a fresh test spectrum with the fault injected, so baseline
noise is included in the false-alarm rate.

| | HPGe | NaI(Tl) |
|---|---|---|
| Channels / gain | 8192 / 0.4 keV | 1024 / 3.0 keV |
| FWHM | sqrt(0.488 + 0.002636 E) → 1.5 keV @ 662, 2.0 keV @ 1332 | 1.3 E^0.55 → 7.0 % @ 662, 5.1 % @ 1332 |
| Check source | Ba-133 + Cs-137 + Co-60 | Am-241 + Cs-137 + Co-60 |
| Reference lines | 303, 356, 662, 1173, 1332 keV (all resolved) | 59.5, 662, and the blended 1173/1332 pair (fitted together) |
| Source run / background run | 600 s / 3600 s | 600 s / 3600 s |

Columns: **caught** = the check that should notice the fault reached WARN or worse; **any WARN+ / any FAIL** = the
overall report, i.e. including false alarms from other checks. Reproduce (deterministic for a given numpy/scipy,
about 2 minutes each):

```bash
gammaguard benchmark --detector hpge --trials 100     # seed 2026
gammaguard benchmark --detector nai  --trials 100
python validation/null_false_alarms.py --trials 400   # healthy-only false-alarm runs
```

## HPGe

```text
case                                    mode        expected check        caught  any WARN+  any FAIL
-----------------------------------------------------------------------------------------------------
healthy                                 source      -                          -         0%        0%
gain drift +0.01 %                      source      calibration               0%         0%        0%
gain drift +0.03 %                      source      calibration             100%       100%        0%
gain drift +0.10 %                      source      calibration             100%       100%      100%
zero shift +0.05 keV                    source      calibration               0%         0%        0%
zero shift +0.30 keV                    source      calibration             100%       100%        1%
zero shift +1.00 keV                    source      calibration             100%       100%      100%
resolution x1.04                        source      resolution                0%         0%        0%
resolution x1.10                        source      resolution               71%        71%        0%
resolution x1.25                        source      resolution              100%       100%      100%
spurious 800 keV, 0.05 cps (30 ct)      source      unexpected_peaks          0%         0%        0%
spurious 800 keV, 0.20 cps (120 ct)     source      unexpected_peaks         25%        25%        0%
spurious 800 keV, 1.00 cps (600 ct)     source      unexpected_peaks         98%        98%        0%
hot channel, +30 counts                 source      channel_anomalies        52%        52%       52%
hot channel, +300 counts                source      channel_anomalies       100%       100%      100%
dead channel at 400 keV (~37 ct/ch)     source      channel_anomalies        99%        99%       99%
dead channel at 880 keV (~10 ct/ch)     source      channel_anomalies         0%         0%        0%
dead time 35 %                          source      acquisition             100%       100%      100%
healthy (background run)                background  -                          -         0%        0%
background x1.10                        background  background                0%         0%        0%
background x1.50                        background  background              100%       100%        0%
background x2.50                        background  background              100%       100%      100%
radon progeny x3                        background  background              100%       100%      100%
electronic noise 0.5 cps                background  background              100%       100%      100%
spurious 700 keV, 0.01 cps (36 ct)      background  unexpected_peaks          0%         0%        0%
spurious 700 keV, 0.03 cps (108 ct)     background  unexpected_peaks         56%        56%        0%

100 trials per case; each trial uses a fresh baseline and a fresh spectrum.
```

## NaI(Tl)

```text
case                                    mode        expected check        caught  any WARN+  any FAIL
-----------------------------------------------------------------------------------------------------
healthy                                 source      -                          -         0%        0%
gain drift +0.2 %                       source      calibration               0%         0%        0%
gain drift +0.8 %                       source      calibration              97%        97%        0%
gain drift +2.0 %                       source      calibration             100%       100%      100%
zero shift +1 keV                       source      calibration               0%         0%        0%
zero shift +8 keV                       source      calibration             100%       100%      100%
zero shift +20 keV                      source      calibration             100%       100%      100%
resolution x1.04                        source      resolution                0%         0%        0%
resolution x1.12                        source      resolution               89%        89%        0%
resolution x1.30                        source      resolution              100%       100%      100%
spurious 2200 keV, 0.3 cps (180 ct)     source      unexpected_peaks         85%        85%        0%
spurious 2200 keV, 1.5 cps (900 ct)     source      unexpected_peaks         98%        98%        0%
hot channel, +180 counts                source      channel_anomalies        99%        99%       99%
hot channel, +1200 counts               source      channel_anomalies       100%       100%      100%
dead channel at 180 keV (~100 ct/ch)    source      channel_anomalies        99%        99%       99%
dead time 35 %                          source      acquisition             100%       100%      100%
healthy (background run)                background  -                          -         0%        0%
background x1.10                        background  background                0%         0%        0%
background x1.50                        background  background              100%       100%        0%
background x2.50                        background  background              100%       100%      100%
radon progeny x3                        background  background               99%        99%        0%
electronic noise 3 cps                  background  background              100%       100%      100%

100 trials per case; each trial uses a fresh baseline and a fresh spectrum.
```

## How to read it

* **No false alarms on healthy detectors**: 0 % in the benchmark rows, and 0 WARN/FAIL in 1,600 further healthy trials
  (400 per detector and mode, `validation/null_false_alarms.py`). With 0 events in 400, the 95 % upper bound on a
  per-mode false-alarm rate is about 0.75 %.
* **Sub-tolerance changes are not alarms** (HPGe gain +0.01 %, NaI gain +0.2 %, resolution ×1.04, background ×1.10).
  An alarm needs the whole confidence interval beyond the tolerance (design notes §4.7).
* **Detection rises smoothly through the tolerance, it does not step at it.** HPGe resolution ×1.10 is 2 % past the ×1.08 warn
  tolerance and is caught 71 % of the time; NaI ×1.12 (4 % past) is caught 89 %; both are caught every time at ×1.25–1.30.
* **Sparse features are limited by counting statistics, as they must be.** An HPGe line of 120 counts on ~10 counts/channel is
  found 25 % of the time, 600 counts 98 %. A dead channel is invisible where the spectrum is sparse (HPGe, ~10 counts/channel: 0 %)
  and found 99 % of the time at ≥37 counts/channel; `P(0 | mean 10)` is simply not small.
* **NaI is a different problem, not a harder one.** Its peaks are broad, so single-channel faults are easy to see against
  them, while a spurious peak has to be separated from a continuum and from the ~10 baseline features that blanket a
  scintillator spectrum (9-12 in these runs), and the match tolerance is therefore tighter (1 FWHM instead of 2).
* The NaI doublet's tied widths are biased low by 3–6 % relative to the simulator truth (a systematic of fitting two blended
  Gaussians on a step); the bias is the same in baseline and test, so it cancels in the QA comparison.

These numbers describe the simulator, not a field detector. Treat them as a regression baseline for the algorithms and as a
way to tune tolerances; real-data results are in [validation.md](validation.md).
