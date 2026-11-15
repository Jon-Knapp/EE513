"""Photon-transfer fit on simulated Poisson-Gaussian flat fields with a known gain and read noise.

The simulated frames are integers (as a raw file is), carry a fixed 1% pixel response
non-uniformity, and sit on a black level, like the real captures."""
import numpy as np
import pytest

K_TRUE = 0.25          # DN per electron
READ_E = 7.0           # read noise, electrons
BLACK = 528.0
FULL_SCALE = 3567.0    # white minus black, DN
LEVELS_E = [150, 400, 1000, 2000, 3500, 5500, 8000]   # all below 70% of full scale


def frame(rng, electrons, prnu, quantize=True):
    signal = rng.poisson(electrons * prnu)
    dn = BLACK + K_TRUE * signal + rng.normal(0, K_TRUE * READ_E, prnu.shape)
    return np.round(dn) if quantize else dn


@pytest.fixture(scope="module")
def flats():
    rng = np.random.default_rng(1)
    prnu = 1 + 0.01 * rng.standard_normal((256, 256))           # fixed pattern, same in every frame
    return rng, prnu


def ptc_points(noise, rng, prnu, quantize=True, single=False):
    means, var_pair, var_single = [], [], []
    for e in LEVELS_E:
        a, b = frame(rng, e, prnu, quantize), frame(rng, e, prnu, quantize)
        m, v, _ = noise["pair_stats"](a, b, BLACK)
        means.append(m); var_pair.append(v)
        var_single.append(np.var(noise["center_patch"](a), ddof=1))
    return np.array(means), np.array(var_pair), np.array(var_single)


def test_pair_method_recovers_gain(noise, flats):
    m, v, _ = ptc_points(noise, *flats)
    fit = noise["fit_ptc"](m, v, FULL_SCALE)
    assert fit["K"] == pytest.approx(K_TRUE, rel=0.03)
    assert abs(fit["K"] - K_TRUE) < 4 * fit["sig_K"]


def test_pair_method_recovers_read_noise(noise, flats):
    m, v, _ = ptc_points(noise, *flats)
    fit = noise["fit_ptc"](m, v, FULL_SCALE)
    assert fit["read"] == pytest.approx(READ_E, rel=0.25)


def test_single_frame_gain_is_biased_high_by_prnu(noise, flats):
    """A single frame's variance also holds the fixed pattern, (p mu)^2, so its fitted gain is too
    high; the difference of a pair cancels it. This is why the report uses pairs."""
    m, v_pair, v_single = ptc_points(noise, *flats)
    k_pair = noise["fit_ptc"](m, v_pair, FULL_SCALE)["K"]
    k_single = noise["fit_ptc"](m, v_single, FULL_SCALE)["K"]
    assert k_single > 1.05 * k_pair


def test_quantization_correction_removes_rounding_variance(noise):
    """Rounding to whole DN adds 1/12 DN^2 to every variance. On exact model points that include it,
    the corrected fit returns the true read noise and the uncorrected one overstates it."""
    mean = np.array([30.0, 100, 300, 800, 1500, 2400])
    var = K_TRUE * mean + (K_TRUE * READ_E) ** 2 + 1 / 12
    corr = noise["fit_ptc"](mean, var, FULL_SCALE, quantization=True)
    raw = noise["fit_ptc"](mean, var, FULL_SCALE, quantization=False)
    assert corr["K"] == pytest.approx(K_TRUE, rel=1e-9)
    assert corr["read"] == pytest.approx(READ_E, rel=1e-9)
    assert raw["read"] > READ_E * 1.01


def test_rounded_frames_carry_one_twelfth(noise):
    """The 1/12 DN^2 itself: pairs of rounded dark frames with continuous noise of known variance."""
    rng = np.random.default_rng(2)
    sd = 0.6                                                      # DN
    a = np.round(BLACK + rng.normal(0, sd, (512, 512)))
    b = np.round(BLACK + rng.normal(0, sd, (512, 512)))
    _, v, _ = noise["pair_stats"](a, b, BLACK, n=512)
    assert v == pytest.approx(sd ** 2 + 1 / 12, abs=0.01)


def test_fit_ignores_points_above_the_cutoff(noise):
    mean = np.array([100, 500, 1000, 2000, 3000, 3400])
    var = 0.25 * mean + 3.0
    var[-2:] = 1.0                                               # "saturated" points, must be ignored
    fit = noise["fit_ptc"](mean, var, FULL_SCALE, frac=0.70)
    assert fit["K"] == pytest.approx(0.25, rel=1e-9)
    assert list(fit["used"]) == [True, True, True, True, False, False]


def test_fit_needs_four_points(noise):
    with pytest.raises(ValueError):
        noise["fit_ptc"]([100, 200, 300], [25, 50, 75], FULL_SCALE)


def test_dynamic_range_and_knee(noise):
    """The report's dynamic range, and the regime crossover: at mu = sigma_r^2 (electrons) the shot
    and read variances are equal, so SNR = mu / sqrt(2 mu)."""
    assert noise["stops"](3567 / 1.694) == pytest.approx(11.04, abs=0.005)
    read = 6.9
    mu = read ** 2
    assert noise["snr_model"](mu, read) == pytest.approx(mu / np.sqrt(2 * mu))
