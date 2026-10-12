"""Open datasets used for validation. They are **downloaded on request and never redistributed**
with GammaGuard: each keeps its own licence (see ``Dataset.license``)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    key: str
    title: str
    detector: str
    url: str
    filename: str
    md5: str
    size_bytes: int
    license: str
    citation: str
    landing_page: str
    note: str = ""


ETH_RLL = Dataset(
    key="eth_rll",
    title="Laboratory based Spectral Measurement Data of the Swiss Airborne Gamma-ray Spectrometer RLL",
    detector="NaI(Tl), four 10.2 x 10.2 x 40.6 cm crystals (16.8 L), 1024 channels",
    url="https://www.research-collection.ethz.ch/server/api/core/bitstreams/"
        "69b39ee6-de83-45b9-a63c-fcd2180c9a58/content",
    filename="RLL_data_v1_4.zip",
    md5="cec885bfdf10f8bd78f0d425a2b0b9c1",
    size_bytes=693_519,
    license=(
        "In Copyright - Non-Commercial Use Permitted (ETH Research Collection). "
        "The paper is CC BY 4.0."
    ),
    citation=(
        "Breitenmoser, D., Butterweck, G., Kasprzak, M., Yukihara, E. G., Mayer, S.: Experimental and Simulated "
        "Spectral Gamma-Ray Response of a NaI(Tl) Scintillation Detector used in Airborne Gamma-Ray Spectrometry, "
        "Advances in Geosciences, 57, 89-107, https://doi.org/10.5194/adgeo-57-89-2022, 2022. "
        "Data: https://doi.org/10.3929/ethz-b-000528920"
    ),
    landing_page="https://doi.org/10.3929/ethz-b-000528920",
    note="Count RATES (cps) after the spectrometer's onboard spectral linearization, not raw counts.",
)

HPGE_CRM = Dataset(
    key="hpge_crm",
    title="HPGe gamma-ray spectra of certified reference materials with decay-corrected labels "
          "(ORTEC GMX35P4-70, 2013-2022)",
    detector=(
        "HPGe: ORTEC GMX35P4-70 (176 spectra) and Canberra detectors (79), "
        "rebinned to 0.5 keV, 20-2900 keV"
    ),
    url="https://zenodo.org/records/23093533/files/Pub6_dataset_v1.0.zip?download=1",
    filename="Pub6_dataset_v1.0.zip",
    md5="64aa9d4786c3e0a65a0355d563e75d09",
    size_bytes=4_810_260,
    license="CC BY 4.0",
    citation=(
        "Huynh, T. Y. H., Truong, H. N. T.: HPGe gamma-ray spectra of certified reference materials with "
        "decay-corrected labels (ORTEC GMX35P4-70, 2013-2022), Zenodo, v1.0, "
        "https://doi.org/10.5281/zenodo.23093533, 2026."
    ),
    landing_page="https://doi.org/10.5281/zenodo.23093533",
    note="Counts are fractional (rebinned). Energy calibration of every spectrum was refitted by the authors.",
)

DATASETS: dict[str, Dataset] = {d.key: d for d in (ETH_RLL, HPGE_CRM)}
