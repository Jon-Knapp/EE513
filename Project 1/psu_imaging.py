"""
Shared helpers for the EE 513 slide figures and demo notebooks.

One copy of the camera simulation, the color transforms, and the synthetic
targets, so that every lecture shows the same scene moving through the same
pipeline rather than five slightly different reimplementations.

Import it from a lecture's Python/ directory or a demo notebook:

    import sys, pathlib
    sys.path.insert(0, str(next(p for p in pathlib.Path.cwd().resolve().parents
                                if (p / "psu_imaging.py").exists())))
    import psu_imaging as psu

or just call psu_imaging.add_to_path() from a sibling script.

Everything here is deliberately plain NumPy and OpenCV. Nothing is hidden in a
library call that a student would be asked to implement themselves.
"""

from pathlib import Path

import numpy as np

__all__ = [
    "PSU_GREEN", "PSU_FOREST", "PSU_BLUE", "PSU_ACCENT", "PSU_WARM", "PSU_GRAY",
    "COLORCHECKER_SRGB", "COLORCHECKER_NAMES",
    "srgb_to_linear", "linear_to_srgb", "linear_rgb_to_xyz", "xyz_to_lab", "delta_e_76",
    "BLACK_LEVEL", "WHITE_LEVEL", "FULL_WELL", "READ_NOISE", "CHANNEL_GAIN",
    "build_scene", "slanted_edge", "frequency_sweep", "color_chart",
    "apply_lens", "mosaic_rggb", "expose_and_digitize", "prnu_map", "simulate_capture",
    "isp", "preference_tune",
    "rmse", "psnr", "find_images_dir", "find_raw_files", "add_to_path",
    "figure_style",
]

# --------------------------------------------------------------------------
# Palette, matched to the psu Marp theme so figures sit inside the slides
# --------------------------------------------------------------------------

PSU_GREEN = "#6d8d24"
PSU_FOREST = "#213921"
PSU_BLUE = "#008ac1"
PSU_ACCENT = "#cfd82d"
PSU_WARM = "#c2703d"
PSU_GRAY = "#8a8f80"


def figure_style():
    """rcParams shared by every lecture's figures. Call once at import time."""
    return {
        "figure.dpi": 120,
        "savefig.bbox": "tight",
        "font.family": "Helvetica Neue",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": __import__("matplotlib").rcParams["axes.prop_cycle"],
        "svg.fonttype": "none",
        "image.interpolation": "nearest",
    }


# --------------------------------------------------------------------------
# Color
# --------------------------------------------------------------------------

#: The 24 patches of a standard ColorChecker, sRGB 8-bit, in the usual order.
COLORCHECKER_SRGB = np.array([
    [115,  82,  68], [194, 150, 130], [ 98, 122, 157], [ 87, 108,  67],
    [133, 128, 177], [103, 189, 170], [214, 126,  44], [ 80,  91, 166],
    [193,  90,  99], [ 94,  60, 108], [157, 188,  64], [224, 163,  46],
    [ 56,  61, 150], [ 70, 148,  73], [175,  54,  60], [231, 199,  31],
    [187,  86, 149], [  8, 133, 161], [243, 243, 242], [200, 200, 200],
    [160, 160, 160], [122, 122, 121], [ 85,  85,  85], [ 52,  52,  52],
], dtype=float) / 255.0

COLORCHECKER_NAMES = [
    "dark skin", "light skin", "blue sky", "foliage", "blue flower", "bluish green",
    "orange", "purplish blue", "moderate red", "purple", "yellow green", "orange yellow",
    "blue", "green", "red", "yellow", "magenta", "cyan",
    "white", "neutral 8", "neutral 6.5", "neutral 5", "neutral 3.5", "black",
]


def srgb_to_linear(x):
    """sRGB encoded values in [0, 1] to linear light."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    """Linear light in [0, 1] to sRGB encoded values."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


#: sRGB primaries, D65 white. Rows map linear RGB to XYZ.
SRGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])

#: D65 reference white in XYZ, Y normalized to 1.
D65_XYZ = np.array([0.95047, 1.00000, 1.08883])


def linear_rgb_to_xyz(rgb):
    return np.asarray(rgb, dtype=float) @ SRGB_TO_XYZ.T


def xyz_to_lab(xyz, white=D65_XYZ):
    """CIE 1976 L*a*b*."""
    t = np.asarray(xyz, dtype=float) / np.asarray(white, dtype=float)
    delta = 6.0 / 29.0
    f = np.where(t > delta ** 3, np.cbrt(np.clip(t, 1e-12, None)),
                 t / (3 * delta ** 2) + 4.0 / 29.0)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def delta_e_76(lab1, lab2):
    """CIE76 color difference. About 1 is a just-noticeable difference."""
    return np.sqrt(np.sum((np.asarray(lab1) - np.asarray(lab2)) ** 2, axis=-1))


# --------------------------------------------------------------------------
# A camera, simulated. Parameters are typical of a good phone sensor.
# --------------------------------------------------------------------------

BLACK_LEVEL = 256            # 12-bit pedestal, in digital numbers
WHITE_LEVEL = 4095           # saturation, in digital numbers
FULL_WELL = 9000.0           # electrons at saturation
READ_NOISE = 2.8             # electrons RMS
PSF_SIGMA = 0.9              # pixels, a decent lens at a good aperture
CHANNEL_GAIN = np.array([0.55, 1.00, 0.62])   # relative spectral sensitivity


def color_chart(h, w, patches=COLORCHECKER_SRGB, gutter=3):
    """A 6 by 4 chart of linear-light patches."""
    out = np.zeros((h, w, 3), dtype=float)
    for i, patch in enumerate(patches):
        r, c = divmod(i, 6)
        y0, y1 = r * h // 4, (r + 1) * h // 4
        x0, x1 = c * w // 6, (c + 1) * w // 6
        out[y0:y1, x0:x1] = srgb_to_linear(patch)
        out[y0:y0 + gutter, x0:x1] = 0.02
        out[y0:y1, x0:x0 + gutter] = 0.02
    return out


def slanted_edge(h, w, angle_deg=5.0, dark=0.05, light=0.90):
    """A near-vertical edge, tilted so it samples the edge response at sub-pixel steps.

    ISO 12233 wants a few degrees of tilt: enough that each row crosses the edge at a
    different sub-pixel phase, not so much that the projection smears the estimate.
    """
    slope = np.tan(np.deg2rad(angle_deg))
    yy, xx = np.mgrid[0:h, 0:w]
    edge = (xx > (w / 2) + slope * (yy - h / 2)).astype(float)
    return srgb_to_linear(dark + (light - dark) * edge)


def frequency_sweep(h, w, f0=0.025, f1=0.50):
    """A horizontal chirp, in cycles per pixel, for showing resolution limits and aliasing."""
    yy, xx = np.mgrid[0:h, 0:w]
    t = xx / max(xx.max(), 1)
    inst = 2 * np.pi * (f0 + (f1 - f0) * t ** 2)
    return srgb_to_linear(0.5 + 0.45 * np.sin(np.cumsum(inst, axis=1)))


def build_scene(h=600, w=900):
    """The course's standard synthetic scene, as linear radiance, shape (h, w, 3).

    Top half is a color chart, bottom left a slanted edge, bottom right a frequency
    sweep. Those are the three targets Project 1 measures, which is the point.
    """
    scene = np.zeros((h, w, 3), dtype=float)
    half = h // 2
    scene[:half] = color_chart(half, w)
    scene[half:, :w // 2] = slanted_edge(h - half, w // 2)[..., None]
    scene[half:, w // 2:] = frequency_sweep(h - half, w - w // 2)[..., None]
    return scene


def apply_lens(scene, sigma=PSF_SIGMA):
    """Stage 1. A Gaussian stand-in for the point spread function."""
    import cv2
    return cv2.GaussianBlur(np.asarray(scene, dtype=np.float64),
                            ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)


def mosaic_rggb(rgb):
    """Stage 2. Keep one color per pixel in an RGGB color filter array.

    Row 0: R G R G ...   even row, even col -> R
    Row 1: G B G B ...   odd  row, odd  col -> B
    """
    rgb = np.asarray(rgb, dtype=float)
    h, w = rgb.shape[:2]
    out = np.zeros((h, w), dtype=float)
    out[0::2, 0::2] = rgb[0::2, 0::2, 0]
    out[0::2, 1::2] = rgb[0::2, 1::2, 1]
    out[1::2, 0::2] = rgb[1::2, 0::2, 1]
    out[1::2, 1::2] = rgb[1::2, 1::2, 2]
    return out


#: Seed for the fixed-pattern maps. These describe the *sensor*, so they must be
#: identical from frame to frame; that is the whole point of "fixed pattern".
FPN_SEED = 20260513


def prnu_map(shape, sigma, seed=FPN_SEED):
    """Photo-response non-uniformity: a fixed per-pixel gain map, mean 1."""
    return 1.0 + sigma * np.random.default_rng(seed).standard_normal(shape)


def expose_and_digitize(mosaic_linear, rng, full_well=FULL_WELL, read_noise=READ_NOISE,
                        black=BLACK_LEVEL, white=WHITE_LEVEL, exposure=1.0,
                        dark_current=0.0, prnu=0.0, fpn_seed=FPN_SEED):
    """Stages 3 and 4. Photons in, integer digital numbers out.

    Poisson shot noise on the signal, Gaussian read noise on the readout, optional
    dark current and photo-response non-uniformity. This is the model lecture 5 derives.

    `prnu` is a fixed per-pixel gain pattern drawn from `fpn_seed`, not a fresh random
    draw per frame. Drawing it per frame would let it average away over a stack, which is
    exactly what fixed-pattern noise does not do.
    """
    electrons = np.clip(np.asarray(mosaic_linear, dtype=float), 0, None) * full_well * exposure
    if prnu:
        electrons = electrons * prnu_map(electrons.shape, prnu, fpn_seed)
    if dark_current:
        electrons = electrons + dark_current
    signal = rng.poisson(np.clip(electrons, 0, None)).astype(float)
    signal = signal + rng.normal(0.0, read_noise, electrons.shape)

    gain_dn_per_e = (white - black) / full_well          # digital numbers per electron
    dn = black + signal * gain_dn_per_e
    return np.clip(np.round(dn), 0, white).astype(np.uint16)


def simulate_capture(scene=None, rng=None, sigma=PSF_SIGMA, channel_gain=CHANNEL_GAIN,
                     **kwargs):
    """Run a scene through lens, CFA, noise, and quantization. Returns (raw, scene)."""
    if scene is None:
        scene = build_scene()
    if rng is None:
        rng = np.random.default_rng(513)
    blurred = apply_lens(scene, sigma) * np.asarray(channel_gain, dtype=float)
    return expose_and_digitize(mosaic_rggb(blurred), rng, **kwargs), scene


# --------------------------------------------------------------------------
# A minimal image signal processor
# --------------------------------------------------------------------------

def isp(raw, black=BLACK_LEVEL, white=WHITE_LEVEL,
        wb_gains=(1 / 0.55, 1.0, 1 / 0.62), ccm=None,
        demosaic="edge", gamma=True, denoise=None):
    """Raw Bayer in, display-referred RGB out. Returns (output, stages_dict).

    demosaic: "edge" for OpenCV's edge-aware method, "bilinear" for the naive one,
              or "none" to stop after white balance.

    Note the OpenCV Bayer constants are named for the second row and column, so an
    RGGB sensor needs COLOR_BayerBG2BGR. Getting this wrong swaps red and blue and is
    the most common bug in student ISP code.
    """
    import cv2

    stages = {}

    x = (np.asarray(raw, dtype=float) - black) / (white - black)
    x = np.clip(x, 0.0, 1.0)
    stages["1. black level"] = x.copy()

    gains = np.asarray(wb_gains, dtype=float)
    wb = x.copy()
    wb[0::2, 0::2] *= gains[0]
    wb[0::2, 1::2] *= gains[1]
    wb[1::2, 0::2] *= gains[1]
    wb[1::2, 1::2] *= gains[2]
    wb = np.clip(wb, 0.0, 1.0)
    stages["2. white balance"] = wb.copy()

    if demosaic == "none":
        stages["3. demosaic"] = wb.copy()
        rgb = np.dstack([wb] * 3)
    else:
        flag = cv2.COLOR_BayerBG2BGR_EA if demosaic == "edge" else cv2.COLOR_BayerBG2BGR
        as_u16 = np.round(wb * 65535).astype(np.uint16)
        rgb = cv2.demosaicing(as_u16, flag).astype(float)[..., ::-1] / 65535.0
        stages["3. demosaic"] = rgb.copy()

    if denoise is not None:
        rgb = denoise(rgb)
        stages["3b. denoise"] = rgb.copy()

    if ccm is not None:
        rgb = np.clip(rgb @ np.asarray(ccm, dtype=float).T, 0.0, 1.0)
    stages["4. color matrix"] = rgb.copy()

    out = linear_to_srgb(rgb) if gamma else np.clip(rgb, 0, 1)
    stages["5. gamma"] = out.copy()
    return out, stages


def preference_tune(img, contrast=1.25, saturation=1.35, sharpen=0.6):
    """A crude imitation of consumer ISP taste: more contrast, saturation, and sharpening."""
    import cv2
    x = np.clip(np.asarray(img, dtype=float), 0, 1)
    x = np.clip(0.5 + contrast * (x - 0.5), 0, 1)
    gray = x.mean(axis=2, keepdims=True)
    x = np.clip(gray + saturation * (x - gray), 0, 1)
    blur = cv2.GaussianBlur(x, (0, 0), 1.2)
    return np.clip(x + sharpen * (x - blur), 0, 1)


# --------------------------------------------------------------------------
# Metrics and file helpers
# --------------------------------------------------------------------------

def rmse(a, b):
    a, b = np.clip(a, 0, 1), np.clip(b, 0, 1)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def psnr(a, b, peak=1.0):
    mse = np.mean((np.clip(a, 0, 1) - np.clip(b, 0, 1)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(peak ** 2 / mse))


RAW_SUFFIXES = {".dng", ".arw", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".pef", ".srw"}


def find_images_dir(start=None):
    """Walk up from `start` (default: the working directory) looking for Images/."""
    here = Path(start or Path.cwd()).resolve()
    for parent in [here, *here.parents]:
        candidate = parent / "Images"
        if candidate.is_dir():
            return candidate
    return None


def find_raw_files(start=None):
    images = find_images_dir(start)
    if images is None:
        return []
    return [p for p in sorted(images.rglob("*"))
            if p.is_file() and p.suffix.lower() in RAW_SUFFIXES]


def add_to_path(start=None):
    """Put the directory holding this module on sys.path. For use from notebooks."""
    import sys
    here = Path(start or Path.cwd()).resolve()
    for parent in [here, *here.parents]:
        if (parent / "psu_imaging.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    raise FileNotFoundError("psu_imaging.py not found above " + str(here))
