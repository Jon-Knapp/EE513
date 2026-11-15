"""Shared fixtures. The analysis functions under test live in two places: camchar.py, and the
homework notebooks (loaded by name with camchar.notebook_functions, so the tests exercise the
exact code the notebooks ran). None of the tests needs the raw captures."""
import sys
from pathlib import Path

import numpy as np
import pytest

P1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(P1))

from camchar import notebook_functions  # noqa: E402


@pytest.fixture(scope="session")
def hw2():
    """Slanted-edge MTF from the Homework 2 notebook."""
    return notebook_functions(P1 / "Measure Your Cameras Spatial Resolution.ipynb",
                              ("slanted_edge_mtf", "mtf_at"), {"np": np})


@pytest.fixture(scope="session")
def noise():
    """Photon-transfer functions from the Homework 3 notebook (same constants as the notebook)."""
    return notebook_functions(P1 / "Noise_and_Linear_Light.ipynb",
                              ("center_patch", "pair_stats", "fit_ptc", "stops", "snr_model"),
                              {"np": np, "PATCH": 128, "Q_VAR": 1 / 12})


@pytest.fixture(scope="session")
def color():
    """White balance, color matrix and chart-orientation check from the Homework 4 notebook."""
    return notebook_functions(P1 / "Color and Accuracy.ipynb",
                              ("wb_gains", "fit_ccm", "check_orientation"),
                              {"np": np, "GAIN_PATCH": 22})
