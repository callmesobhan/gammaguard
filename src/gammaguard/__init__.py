"""GammaGuard: anomaly detection and QA for gamma-ray spectrometers (HPGe and NaI(Tl))."""

from .calibration import EnergyCalibration
from .qa import HPGE, NAI, Baseline, DetectorProfile, QAReport, Status, Tolerances, run_qa
from .resolution import PowerLawResolution, ResolutionModel
from .spectrum import Spectrum

__version__ = "0.2.0"

__all__ = [
    "HPGE", "NAI", "Baseline", "DetectorProfile", "EnergyCalibration", "PowerLawResolution", "QAReport",
    "ResolutionModel", "Spectrum", "Status", "Tolerances", "run_qa",
]
