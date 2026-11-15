"""camchar: square finder, DNG opcode parsing, Apple's warp factor, distortion profile,
straightness measure and the bootstrap helper."""
import json
import struct

import cv2
import numpy as np
import pytest

import camchar


# ----------------------------------------------------------------------------- square finder
def target_image(h=1500, w=2000, falloff=0.12, seed=0):
    """Black squares (5% reflectance) on paper lit very unevenly: the paper drops to `falloff` of
    its peak brightness toward one corner, as under the ring light that broke a global threshold."""
    yy, xx = np.mgrid[0:h, 0:w].astype(float)
    paper = 1 - (1 - falloff) * np.hypot(xx / w, yy / h) / np.sqrt(2)
    reflect = np.ones((h, w))
    centers = [(300, 300), (1000, 400), (1700, 300), (500, 1100), (1600, 1150)]
    for cx, cy in centers:
        reflect[cy - 70:cy + 70, cx - 70:cx + 70] = 0.05
    rng = np.random.default_rng(seed)
    return 3000 * paper * reflect + rng.normal(0, 5, (h, w)), centers


def test_square_finder_handles_uneven_light():
    img, centers = target_image()
    found = camchar.find_squares_local(img)
    assert len(found) == len(centers)
    for cx, cy in centers:
        d = min(np.hypot(*(s["center"] - (cx, cy))) for s in found)
        assert d < 3
    assert all(abs(s["side"] - 140) < 6 for s in found)


def test_square_finder_rejects_squares_cut_by_the_frame():
    img, centers = target_image()
    img[0:150, 1900:2000] = 3000 * 0.05                          # a dark square running off the corner
    found = camchar.find_squares_local(img)
    assert len(found) == len(centers)


def test_global_threshold_would_fail_here():
    """Why the local finder exists: one global Otsu threshold merges the dim paper with the squares."""
    img, _ = target_image()
    small = cv2.resize(img.astype(np.float32), (500, 375), interpolation=cv2.INTER_AREA)
    u8 = cv2.normalize(small, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, bw = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    assert bw.mean() / 255 > 0.15                                # far more "dark" area than 5 squares


def test_multi_window_finder_does_not_double_count():
    img, centers = target_image()
    assert len(camchar.find_squares_multi(img)) == len(centers)


# ----------------------------------------------------------------------------- DNG opcodes
def opcode(oid, payload):
    return struct.pack(">IIII", oid, 0x01030000, 1, len(payload)) + payload


def test_parse_opcode_list_reads_warp_coefficients():
    kr = [1.0, 0.0, 0.1376, 0.0, -0.7529, 0.0, 1.4557, 0.0, -1.1507, 0.0, 0.3952, 0.0, -0.0127, 0.0, -0.0193]
    warp = struct.pack(">I", 1) + struct.pack(">15d", *kr) + struct.pack(">6d", 0, 0, 0, 1, 0.5, 0.5)
    vignette = struct.pack(">7d", 1, -0.5, 0.2, 0, 0, 0.5, 0.5)
    blob = struct.pack(">I", 2) + opcode(3, vignette) + opcode(14, warp)
    ops = camchar.parse_opcode_list(blob)
    assert [o["opcode"] for o in ops] == ["FixVignetteRadial", "WarpRectilinear2"]
    assert ops[1]["radial_coefficients"] == pytest.approx(kr)


def test_parse_opcode_list_names_unknown_opcodes_by_id():
    blob = struct.pack(">I", 1) + opcode(9, b"\x00" * 8)
    assert camchar.parse_opcode_list(blob) == [{"opcode": 9}]


def test_f_apple_is_the_radial_polynomial():
    kr = np.array([1.0, 0, 0.1, 0, -0.2] + [0.0] * 10)
    r = np.array([0.0, 0.5, 1.0])
    assert camchar.f_apple(r, kr) == pytest.approx(1 + 0.1 * r ** 2 - 0.2 * r ** 4)


# ----------------------------------------------------------------------------- distortion profile
def test_radial_profile_matches_the_model():
    """One radial term, k1: a raw radius r_d (in focal lengths) comes from r_u with
    r_d = r_u (1 + k1 r_u^2), so the displacement is 100 (r_d / r_u - 1) percent."""
    f, k1, corner_r = 1000.0, 0.1, 1000.0
    K = np.array([[f, 0, 2000], [0, f, 1500], [0, 0, 1]])
    d = np.array([k1, 0, 0, 0, 0], float)
    fracs = [0.25, 0.5, 0.83]
    got = camchar.radial_profile(K, d, fracs, corner_r)
    for fr, g in zip(fracs, got):
        rd = fr * corner_r / f
        ru = min(r.real for r in np.roots([k1, 0, 1, -rd]) if abs(r.imag) < 1e-12 and r.real > 0)
        assert g == pytest.approx(100 * (rd / ru - 1), abs=1e-3)


def test_radial_profile_is_zero_without_distortion():
    K = np.array([[1500.0, 0, 2016], [0, 1500.0, 1512], [0, 0, 1]])
    assert np.allclose(camchar.radial_profile(K, np.zeros(5), [0.3, 0.8], 2520.0), 0, atol=1e-4)


# ----------------------------------------------------------------------------- straightness
def test_bow_of_a_tilted_straight_line_is_zero():
    x = np.linspace(0, 4000, 1000)
    assert camchar.bow(x, 2900 + 0.013 * x)[0] == pytest.approx(0, abs=1e-9)


def test_bow_of_a_parabola():
    """y = a x^2 on [-L, L]: a straight-line fit removes only the mean, so peak-to-valley = a L^2."""
    x = np.linspace(-1000, 1000, 2001)
    pv, _ = camchar.bow(x, 5 + 0.01 * x + 1e-5 * x ** 2)
    assert pv == pytest.approx(10.0, rel=1e-6)


def test_bow_noise_measures_edge_scatter():
    rng = np.random.default_rng(6)
    x = np.linspace(0, 4000, 1000)
    y = 1e-6 * (x - 2000) ** 2 + rng.normal(0, 0.4, x.size)
    out = camchar.bow_noise(x, y)
    assert out["edge_noise_rms_px"] == pytest.approx(0.4 * np.sqrt(10 / 11), rel=0.1)
    pv_points = camchar.bow(x, y)[0]
    # noise inflates the peak-to-valley of the points; the running mean removes most of that
    assert pv_points > out["smooth_pv_px"] > 3.8
    assert abs(out["smooth_pv_px"] - 4.0) < abs(pv_points - 4.0)


def test_median_se_matches_large_sample_theory():
    """For normal data the median's standard error is about 1.2533 sigma / sqrt(n). One sample's
    bootstrap estimate scatters around that, so average over 20 samples."""
    rng = np.random.default_rng(7)
    est = [camchar.median_se(rng.normal(0, 1, 400), n_boot=500) for _ in range(20)]
    assert np.mean(est) == pytest.approx(1.2533 / np.sqrt(400), rel=0.1)


# ----------------------------------------------------------------------------- notebook loader
def test_notebook_functions_loads_and_rejects(tmp_path):
    nb = {"cells": [{"cell_type": "code", "source": ["X = 3\n", "def triple(v):\n", "    return X * v\n"]},
                    {"cell_type": "markdown", "source": ["def not_code(): pass\n"]}]}
    path = tmp_path / "nb.ipynb"
    path.write_text(json.dumps(nb))
    ns = camchar.notebook_functions(path, ["triple"], {"X": 2})
    assert ns["triple"](5) == 10                                  # uses the namespace, not the cell's X
    with pytest.raises(KeyError):
        camchar.notebook_functions(path, ["not_code"])
