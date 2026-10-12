# Third-party data and licences

GammaGuard's **code** is licensed under the Apache License 2.0 (see [LICENSE](../LICENSE) and [NOTICE](../NOTICE)). The real spectra used for validation are
third-party datasets with their **own licences**. They are **not included** in this repository or in the package: they are
downloaded on request, checksum-verified, and cached locally.

```bash
gammaguard fetch-data all        # or: eth_rll | hpge_crm.  Cache: $GAMMAGUARD_DATA or ~/.cache/gammaguard
```

| Key | Dataset | Detector | Licence | Size |
|---|---|---|---|---|
| `eth_rll` | [Laboratory based Spectral Measurement Data of the Swiss Airborne Gamma-ray Spectrometer RLL](https://doi.org/10.3929/ethz-b-000528920) (ETH Research Collection) | NaI(Tl), 4 × (10.2 × 10.2 × 40.6 cm), 1024 ch | **In Copyright – Non-Commercial Use Permitted**. The accompanying paper is CC BY 4.0. | 0.7 MB |
| `hpge_crm` | [HPGe gamma-ray spectra of certified reference materials (ORTEC GMX35P4-70, 2013–2022)](https://doi.org/10.5281/zenodo.23093533) (Zenodo) | HPGe, ORTEC GMX35P4-70 and Canberra | CC BY 4.0 | 4.8 MB |

**Why the data are not bundled.** The ETH dataset is licensed for non-commercial use only, which is not compatible with
redistribution inside a permissively licensed (here Apache-2.0) package. Fetching it yourself, for research, is within its terms; check them before
any commercial use. The summary statistics in `validation/results/*.json` are derived numbers (centroids, widths, rates), not
the spectra.

**Please cite** the data if you use the validation results:

* Breitenmoser, D., Butterweck, G., Kasprzak, M., Yukihara, E. G., Mayer, S.: *Experimental and Simulated Spectral Gamma-Ray
  Response of a NaI(Tl) Scintillation Detector used in Airborne Gamma-Ray Spectrometry*, Advances in Geosciences, 57, 89–107,
  <https://doi.org/10.5194/adgeo-57-89-2022>, 2022. Data: <https://doi.org/10.3929/ethz-b-000528920>.
* Huynh, T. Y. H., Truong, H. N. T.: *HPGe gamma-ray spectra of certified reference materials with decay-corrected labels
  (ORTEC GMX35P4-70, 2013–2022)*, Zenodo, v1.0, <https://doi.org/10.5281/zenodo.23093533>, 2026.

Downloads are verified against the md5 sums published by the repositories (`src/gammaguard/datasets/registry.py`), archive
members are checked for path traversal before extraction, and the NumPy archive is loaded with `allow_pickle=False`.

The nuclide data in `src/gammaguard/isotopes.py` are illustrative (transcribed from memory); verify against NNDC / LNHB / DDEP
before using them beyond simulation and labelling.
