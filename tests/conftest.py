import numpy as np
import pytest

from gammaguard import Baseline
from gammaguard.simulate import background_scenario, simulate, source_scenario


@pytest.fixture(scope="session")
def baseline() -> Baseline:
    rng = np.random.default_rng(12345)
    return Baseline.build(simulate(source_scenario(1800.0), rng), simulate(background_scenario(4 * 3600.0), rng))


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(2026)
