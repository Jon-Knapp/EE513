"""White balance, color correction matrix and chart-orientation check."""
import colour
import numpy as np
import pytest

REF = colour.CCS_COLOURCHECKERS["ColorChecker24 - After November 2014"]
REF_XYZ = np.array([colour.xyY_to_XYZ(v) for v in REF.data.values()])


@pytest.fixture(scope="module")
def chart_rgb():
    """A camera-like linear RGB reading of the reference chart: the reference XYZ seen through the
    sRGB primaries, then multiplied by an illuminant-like cast that white balance must undo."""
    rgb = colour.XYZ_to_RGB(REF_XYZ, colour.RGB_COLOURSPACES["sRGB"], illuminant=REF.illuminant)
    return np.clip(rgb, 1e-4, None) * np.array([0.55, 1.0, 1.8])


def test_wb_gains_neutralize_the_gain_patch(color, chart_rgb):
    g = color["wb_gains"](chart_rgb)
    assert g[1] == pytest.approx(1.0)                            # normalized to green
    balanced = chart_rgb[21] * g                                 # patch 22, neutral 5
    assert np.allclose(balanced, balanced[1], rtol=1e-12)


def test_wb_gains_undo_a_channel_cast(color, chart_rgb):
    cast = np.array([2.0, 1.0, 0.5])
    g0 = color["wb_gains"](chart_rgb)
    g1 = color["wb_gains"](chart_rgb * cast)
    assert np.allclose(g1, g0 / cast)


def test_ccm_recovers_a_known_matrix(color):
    rng = np.random.default_rng(3)
    M = np.array([[1.2, 1.1, 0.13], [0.03, 2.8, -0.21], [0.09, -1.4, 3.5]])
    c = rng.uniform(0.02, 0.9, (24, 3))
    t = c @ M.T
    fit = color["fit_ccm"](c, t, np.ones(24, bool))
    assert np.allclose(fit, M, atol=1e-10)


def test_ccm_ignores_excluded_patches(color):
    rng = np.random.default_rng(4)
    M = np.diag([1.0, 2.0, 3.0])
    c = rng.uniform(0.02, 0.9, (24, 3))
    t = c @ M.T
    t[18] += 10.0                                                # a "clipped" patch with a wrong value
    use = np.ones(24, bool); use[18] = False
    assert np.allclose(color["fit_ccm"](c, t, use), M, atol=1e-10)


def test_least_squares_is_optimal(color):
    """No small change to the fitted matrix lowers the summed squared error."""
    rng = np.random.default_rng(5)
    c = rng.uniform(0.02, 0.9, (24, 3))
    t = rng.uniform(0.02, 0.9, (24, 3))
    use = np.ones(24, bool)
    M = color["fit_ccm"](c, t, use)
    err = lambda m: ((c @ m.T - t) ** 2).sum()
    for _ in range(20):
        assert err(M) <= err(M + 1e-4 * rng.standard_normal((3, 3)))


def test_orientation_check_accepts_the_chart(color, chart_rgb):
    assert color["check_orientation"](chart_rgb) == []


@pytest.mark.parametrize("transform", ["rotated", "mirrored"])
def test_orientation_check_rejects_a_misread_chart(color, chart_rgb, transform):
    grid = chart_rgb.reshape(4, 6, 3)
    grid = grid[::-1, ::-1] if transform == "rotated" else grid[:, ::-1]
    assert color["check_orientation"](grid.reshape(24, 3)) != []
