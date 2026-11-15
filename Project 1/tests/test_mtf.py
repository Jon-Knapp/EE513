"""Slanted-edge MTF against an edge whose MTF is known exactly.

A straight edge blurred by a Gaussian of standard deviation sigma has
MTF(f) = exp(-2 pi^2 sigma^2 f^2), so MTF50 = sqrt(ln 2 / 2) / (pi sigma) cycles per pixel."""
import numpy as np
import pytest
from scipy.special import erf


def gaussian_edge(sigma, angle_deg=5.0, h=120, w=96, lo=0.1, hi=0.9):
    """A dark-to-light edge a few degrees off vertical, point-sampled at the pixel centers."""
    y, x = np.mgrid[0:h, 0:w].astype(float)
    a = np.radians(angle_deg)
    d = (x - w / 2 - (y - h / 2) * np.tan(a)) * np.cos(a)          # signed distance to the edge
    return lo + (hi - lo) * 0.5 * (1 + erf(d / (sigma * np.sqrt(2))))


def mtf50_exact(sigma):
    return np.sqrt(np.log(2) / 2) / (np.pi * sigma)


@pytest.mark.parametrize("sigma", [0.8, 1.0, 1.5, 2.0])
def test_mtf50_of_gaussian_edge(hw2, sigma):
    res = hw2["slanted_edge_mtf"](gaussian_edge(sigma))
    assert res["mtf50"] == pytest.approx(mtf50_exact(sigma), rel=0.01)


def test_whole_curve_matches_exact_mtf(hw2):
    res = hw2["slanted_edge_mtf"](gaussian_edge(1.0))
    f = np.array([0.05, 0.1, 0.2, 0.3])
    exact = np.exp(-2 * np.pi ** 2 * f ** 2)
    measured = np.array([hw2["mtf_at"](res, fi) for fi in f])
    assert np.allclose(measured, exact, atol=0.01)
    assert res["mtf"][0] == pytest.approx(1.0)


def test_near_horizontal_edge_gives_same_answer(hw2):
    """The function transposes a near-horizontal edge; the answer must not depend on it."""
    edge = gaussian_edge(1.2)
    vertical = hw2["slanted_edge_mtf"](edge)
    horizontal = hw2["slanted_edge_mtf"](edge.T)
    assert horizontal["transposed"] and not vertical["transposed"]
    assert horizontal["mtf50"] == pytest.approx(vertical["mtf50"], rel=1e-6)


def test_edge_angle_is_measured(hw2):
    res = hw2["slanted_edge_mtf"](gaussian_edge(1.0, angle_deg=6.0))
    assert abs(res["angle_deg"]) == pytest.approx(6.0, abs=0.1)


def test_dark_on_light_edge_is_flipped(hw2):
    """A falling edge gives the same MTF as a rising one."""
    edge = gaussian_edge(1.0)
    up = hw2["slanted_edge_mtf"](edge)
    down = hw2["slanted_edge_mtf"](edge[:, ::-1])
    assert down["mtf50"] == pytest.approx(up["mtf50"], rel=0.01)
