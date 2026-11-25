"""GammaGuard: anomaly detection and QA for gamma-ray spectrometers (HPGe and NaI(Tl))."""

from .calibration import EnergyCalibration
from .qa import (
    HPGE,
    NAI,
    Baseline,
    DetectorProfile,
    HardwareDiagnosis,
    QAReport,
    Status,
    Tolerances,
    TrendForecast,
    analyze_trend_series,
    diagnose_hardware,
    run_qa,
)
from .resolution import PowerLawResolution, ResolutionModel
from .spectrum import Spectrum

__version__ = "0.3.0"

__all__ = [
    "HPGE", "NAI", "Baseline", "DetectorProfile", "EnergyCalibration", "HardwareDiagnosis",
    "PowerLawResolution", "QAReport", "ResolutionModel", "Spectrum", "Status", "Tolerances",
    "TrendForecast", "analyze_trend_series", "diagnose_hardware", "run_qa",
]
