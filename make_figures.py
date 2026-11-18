"""
Figures for EE 513 lecture 7: The image signal processor.

Includes bilinear and Malvar-He-Cutler demosaicing written out, since implementing
both is a stated outcome and Project 2 asks for them.

    python make_figures.py
"""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import psu_imaging as psu                                       # noqa: E402

FIGURES = HERE / "Figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update(psu.figure_style())
np.seterr(divide="ignore", over="ignore", invalid="ignore")


#: Real colour filter arrays have broad, overlapping passbands, so the camera's RGB is a
#: mixture of the scene's. Without this the white-balance gains alone would already be
#: exact and the colour matrix would come out as the identity, which is not the world.
CFA_CROSSTALK = np.array([
    [0.82, 0.22, -0.04],
    [0.11, 0.79, 0.10],
    [0.02, 0.19, 0.79],
])


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.svg")
    plt.close(fig)
    print(f"wrote Figures/{name}.svg")


# ==========================================================================
# Demosaicing, written out
# ==========================================================================

def bilinear_demosaic(bayer):
    """Bilinear interpolation on an RGGB mosaic. The first thing anyone writes."""
    h, w = bayer.shape
    r = np.zeros((h, w)); g = np.zeros((h, w)); b = np.zeros((h, w))
    r[0::2, 0::2] = bayer[0::2, 0::2]
    g[0::2, 1::2] = bayer[0::2, 1::2]
    g[1::2, 0::2] = bayer[1::2, 0::2]
    b[1::2, 1::2] = bayer[1::2, 1::2]

    # Known-sample masks, so we divide by how many neighbors actually existed.
    mr = np.zeros((h, w)); mr[0::2, 0::2] = 1
    mg = np.zeros((h, w)); mg[0::2, 1::2] = 1; mg[1::2, 0::2] = 1
    mb = np.zeros((h, w)); mb[1::2, 1::2] = 1

    k_rb = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float) / 4.0
    k_g = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=float) / 4.0

    def norm(plane, mask, kernel):
        num = cv2.filter2D(plane, -1, kernel, borderType=cv2.BORDER_REFLECT)
        den = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
        return num / np.maximum(den, 1e-9)

    return np.dstack([norm(r, mr, k_rb), norm(g, mg, k_g), norm(b, mb, k_rb)])


def mhc_demosaic(bayer):
    """Malvar, He and Cutler (2004): bilinear plus a gradient correction term.

    The idea is that the luminance detail missing from one channel is visible in the
    channel that *was* sampled at that pixel, so a Laplacian of the measured channel
    corrects the interpolation. Five linear filters, no branching, still the baseline
    in a lot of production code.
    """
    h, w = bayer.shape
    out = bilinear_demosaic(bayer)

    # The five MHC kernels, scaled by 1/8 as published.
    g_at_rb = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0]], dtype=float) / 8.0
    rb_at_g_rrow = np.array([
        [0, 0, 0.5, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 0.5, 0, 0]], dtype=float) / 8.0
    rb_at_g_crow = rb_at_g_rrow.T
    rb_at_br = np.array([
        [0, 0, -1.5, 0, 0],
        [0, 2, 0, 2, 0],
        [-1.5, 0, 6, 0, -1.5],
        [0, 2, 0, 2, 0],
        [0, 0, -1.5, 0, 0]], dtype=float) / 8.0

    f = lambda k: cv2.filter2D(bayer, -1, k, borderType=cv2.BORDER_REFLECT)
    g_est, rg_r, rg_c, rb_est = f(g_at_rb), f(rb_at_g_rrow), f(rb_at_g_crow), f(rb_at_br)

    # Green at red and blue sites.
    out[0::2, 0::2, 1] = g_est[0::2, 0::2]
    out[1::2, 1::2, 1] = g_est[1::2, 1::2]

    # Red at green sites: red varies along the row on red rows, along the column on blue rows.
    out[0::2, 1::2, 0] = rg_r[0::2, 1::2]
    out[1::2, 0::2, 0] = rg_c[1::2, 0::2]
    # Blue at green sites: the other way around.
    out[0::2, 1::2, 2] = rg_c[0::2, 1::2]
    out[1::2, 0::2, 2] = rg_r[1::2, 0::2]

    # Red at blue sites and blue at red sites.
    out[1::2, 1::2, 0] = rb_est[1::2, 1::2]
    out[0::2, 0::2, 2] = rb_est[0::2, 0::2]

    return np.clip(out, 0, 1)


def opencv_demosaic(bayer, edge_aware=True):
    u16 = np.round(np.clip(bayer, 0, 1) * 65535).astype(np.uint16)
    flag = cv2.COLOR_BayerBG2BGR_EA if edge_aware else cv2.COLOR_BayerBG2BGR
    return cv2.demosaicing(u16, flag).astype(float)[..., ::-1] / 65535.0


# ==========================================================================

def fig_isp_chain():
    stages = ["black\nlevel", "lens\nshading", "defective\npixels", "demosaic",
              "denoise", "white\nbalance", "color\nmatrix", "tone\nmap", "sharpen", "encode"]
    fig, ax = plt.subplots(figsize=(13, 2.2))
    ax.set_axis_off(); ax.set_xlim(0, len(stages) * 1.32); ax.set_ylim(-0.9, 0.9)
    for i, name in enumerate(stages):
        x = i * 1.32
        linear = i < 7
        ax.add_patch(FancyBboxPatch((x, -0.3), 1.05, 0.62,
                                    boxstyle="round,pad=0.03,rounding_size=0.06",
                                    lw=1.8,
                                    edgecolor=psu.PSU_GREEN if linear else psu.PSU_WARM,
                                    facecolor="#f4f7ec" if linear else "#fbf1ea"))
        ax.text(x + 0.52, 0.01, name, ha="center", va="center", fontsize=8.5,
                color=psu.PSU_FOREST)
        if i < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.08, 0.01), (x + 1.28, 0.01),
                                         arrowstyle="-|>", mutation_scale=9,
                                         lw=1.2, color="#888"))
    ax.plot([0, 9.3], [-0.52, -0.52], lw=2.5, color=psu.PSU_GREEN)
    ax.text(4.6, -0.75, "linear light", ha="center", fontsize=9.5, color=psu.PSU_GREEN)
    ax.plot([9.35, 13.1], [-0.52, -0.52], lw=2.5, color=psu.PSU_WARM)
    ax.text(11.2, -0.75, "display referred", ha="center", fontsize=9.5, color=psu.PSU_WARM)
    save(fig, "L07-ISPChain")


def _capture(sigma=0.8, seed=3):
    scene = psu.build_scene()
    blurred = psu.apply_lens(scene, sigma) * psu.CHANNEL_GAIN
    raw = psu.expose_and_digitize(psu.mosaic_rggb(blurred), np.random.default_rng(seed))
    norm = (raw.astype(float) - psu.BLACK_LEVEL) / (psu.WHITE_LEVEL - psu.BLACK_LEVEL)
    gains = np.array([1 / 0.55, 1.0, 1 / 0.62])
    wb = np.clip(norm, 0, 1).copy()
    wb[0::2, 0::2] *= gains[0]; wb[0::2, 1::2] *= gains[1]
    wb[1::2, 0::2] *= gains[1]; wb[1::2, 1::2] *= gains[2]
    return np.clip(wb, 0, 1), scene


def fig_demosaic_compare():
    wb, scene = _capture()
    truth = psu.linear_to_srgb(scene)

    methods = {
        "Bilinear": bilinear_demosaic(wb),
        "Malvar-He-Cutler": mhc_demosaic(wb),
        "OpenCV edge aware": opencv_demosaic(wb),
    }

    ry, rx, rn = 320, 480, 120          # into the frequency sweep, where it hurts
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (name, img) in zip(axes, methods.items()):
        srgb = psu.linear_to_srgb(img)
        ax.imshow(srgb[ry:ry + rn, rx:rx + rn])
        ax.set_title(f"{name}\nPSNR {psu.psnr(srgb, truth):.1f} dB", fontsize=9.5)
    axes[3].imshow(truth[ry:ry + rn, rx:rx + rn]); axes[3].set_title("Ground truth", fontsize=9.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("False color where the detail is finer than the color filter array",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save(fig, "L07-DemosaicCompare")
    for name, img in methods.items():
        print(f"  {name:20s} PSNR {psu.psnr(psu.linear_to_srgb(img), truth):5.2f} dB")


def fig_zippering():
    wb, scene = _capture()
    truth = psu.linear_to_srgb(scene)
    bil = psu.linear_to_srgb(bilinear_demosaic(wb))
    mhc = psu.linear_to_srgb(mhc_demosaic(wb))

    ry, rx, rn = 330, 200, 64            # across the slanted edge
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2))
    for j, (img, name) in enumerate([(bil, "Bilinear"), (mhc, "Malvar-He-Cutler"),
                                     (truth, "Ground truth")]):
        axes[0, j].imshow(img[ry:ry + rn, rx:rx + rn])
        axes[0, j].set_title(name, fontsize=10)
        err = np.abs(img - truth).mean(axis=2)
        axes[1, j].imshow(err[ry:ry + rn, rx:rx + rn], cmap="magma", vmin=0, vmax=0.15)
        axes[1, j].set_title("error" if j < 2 else "(zero by definition)", fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Zippering along a near-vertical edge: alternating rows guess differently",
                 fontsize=10.5, y=1.01)
    fig.tight_layout()
    save(fig, "L07-Zippering")


def fig_order_matters():
    """Four orderings, one of which is right."""
    wb_mosaic, scene = _capture()
    truth = psu.linear_to_srgb(scene)

    # Rebuild the unbalanced mosaic so we can reorder white balance and demosaic.
    blurred = (psu.apply_lens(scene, 0.8) @ CFA_CROSSTALK.T) * psu.CHANNEL_GAIN
    raw = psu.expose_and_digitize(psu.mosaic_rggb(blurred), np.random.default_rng(3))
    mosaic = np.clip((raw.astype(float) - psu.BLACK_LEVEL)
                     / (psu.WHITE_LEVEL - psu.BLACK_LEVEL), 0, 1)
    gains = np.array([1 / 0.55, 1.0, 1 / 0.62])

    def wb_mosaic_first(m):
        x = m.copy()
        x[0::2, 0::2] *= gains[0]; x[0::2, 1::2] *= gains[1]
        x[1::2, 0::2] *= gains[1]; x[1::2, 1::2] *= gains[2]
        return np.clip(x, 0, 1)

    # Fit the colour matrix to THIS camera from the chart patches, so that every
    # variant below runs the same pipeline and differs only in the order of its stages.
    lin = mhc_demosaic(wb_mosaic_first(mosaic))
    half = scene.shape[0] // 2
    ph, pw = half // 4, scene.shape[1] // 6
    src, dst = [], []
    for i in range(24):
        r, c = divmod(i, 6)
        y0, x0 = r * ph + ph // 4, c * pw + pw // 4
        sl = (slice(y0, y0 + ph // 2), slice(x0, x0 + pw // 2))
        src.append(lin[sl].reshape(-1, 3).mean(axis=0))
        dst.append(scene[sl].reshape(-1, 3).mean(axis=0))
    CCM, *_ = np.linalg.lstsq(np.array(src), np.array(dst), rcond=None)
    CCM = CCM.T

    right = psu.linear_to_srgb(np.clip(lin @ CCM.T, 0, 1))

    # Black level left in: the pedestal survives into the colour ratios, so everything
    # gets a cast that is worst where the signal is smallest.
    with_pedestal = np.clip(raw.astype(float) / psu.WHITE_LEVEL, 0, 1)
    no_black = psu.linear_to_srgb(
        np.clip(mhc_demosaic(wb_mosaic_first(with_pedestal)) @ CCM.T, 0, 1))

    # Same matrix, applied after the transfer function instead of before it.
    gamma_first = np.clip(psu.linear_to_srgb(lin) @ CCM.T, 0, 1)

    # Sharpen before denoise, on a noisier capture.
    noisy_raw = psu.expose_and_digitize(psu.mosaic_rggb(blurred * 0.12),
                                        np.random.default_rng(9))
    noisy = np.clip((noisy_raw.astype(float) - psu.BLACK_LEVEL)
                    / (psu.WHITE_LEVEL - psu.BLACK_LEVEL), 0, 1) / 0.12
    nimg = np.clip(mhc_demosaic(wb_mosaic_first(np.clip(noisy, 0, 1))) @ CCM.T, 0, 1)
    sharp_first = psu.linear_to_srgb(
        np.clip(cv2.GaussianBlur(
            np.clip(nimg + 0.9 * (nimg - cv2.GaussianBlur(nimg, (0, 0), 1.2)), 0, 1),
            (0, 0), 1.0), 0, 1))

    ry, rx, rn = 60, 60, 150
    panels = [(right, "Correct order"),
              (no_black, "Black level never subtracted"),
              (gamma_first, "Gamma BEFORE the color matrix"),
              (sharp_first, "Sharpen BEFORE denoise")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
    for ax, (img, name) in zip(axes, panels):
        ax.imshow(np.clip(img[ry:ry + rn, rx:rx + rn], 0, 1))
        ax.set_title(name, fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Same stages, different order. Three of these ship in somebody's product.",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save(fig, "L07-OrderMatters")
    print("  fitted colour matrix (camera RGB -> scene RGB):")
    for row in CCM:
        print("    " + "  ".join(f"{v:+.3f}" for v in row))
    for img, name in panels:
        print(f"  {name:32s} PSNR {psu.psnr(np.clip(img, 0, 1), truth):5.2f} dB")

    # The white-balance ordering convention, measured rather than asserted.
    wb_after = psu.linear_to_srgb(np.clip(mhc_demosaic(mosaic) * gains, 0, 1))
    print(f"  {'[wb before demosaic]':32s} PSNR {psu.psnr(right, truth):5.2f} dB")
    print(f"  {'[wb after demosaic]':32s} PSNR {psu.psnr(wb_after, truth):5.2f} dB")


def fig_tone_and_detail():
    """Global tone curves, and the noise-versus-detail trade in denoising."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))

    x = np.linspace(0, 1, 256)
    for gamma, color, label in [(1.0, "#999", "linear"), (1 / 2.2, psu.PSU_FOREST, "sRGB-ish"),
                                (1 / 3.0, psu.PSU_BLUE, "lifted shadows")]:
        axes[0].plot(x, x ** gamma, lw=2, color=color, label=label)
    s = 1 / (1 + np.exp(-8 * (x - 0.5)))
    axes[0].plot(x, (s - s[0]) / (s[-1] - s[0]), lw=2, color=psu.PSU_WARM, label="S-curve, contrast")
    axes[0].set_xlabel("linear light in"); axes[0].set_ylabel("encoded out")
    axes[0].set_title("Global tone curves"); axes[0].legend(fontsize=8.5)

    wb, scene = _capture(seed=11)
    noisy_raw = psu.expose_and_digitize(
        psu.mosaic_rggb(psu.apply_lens(scene, 0.8) * psu.CHANNEL_GAIN * 0.05),
        np.random.default_rng(12))
    noisy = np.clip((noisy_raw.astype(float) - psu.BLACK_LEVEL)
                    / (psu.WHITE_LEVEL - psu.BLACK_LEVEL), 0, 1) / 0.05
    gains = np.array([1 / 0.55, 1.0, 1 / 0.62])
    m = np.clip(noisy, 0, 1).copy()
    m[0::2, 0::2] *= gains[0]; m[0::2, 1::2] *= gains[1]
    m[1::2, 0::2] *= gains[1]; m[1::2, 1::2] *= gains[2]
    img = mhc_demosaic(np.clip(m, 0, 1))

    ry, rx, rn = 300, 100, 120
    strengths = [0, 6, 18]
    for ax, s_ in zip(axes[1:], strengths[1:]):
        out = cv2.fastNlMeansDenoisingColored(
            (np.clip(psu.linear_to_srgb(img), 0, 1) * 255).astype(np.uint8),
            None, s_, s_, 7, 21).astype(float) / 255
        ax.imshow(out[ry:ry + rn, rx:rx + rn])
        ax.set_title(f"denoise strength {s_}", fontsize=9.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Denoising always spends detail. Tuning is choosing how much.",
                 fontsize=10.5, y=1.02)
    fig.tight_layout()
    save(fig, "L07-ToneAndDetail")


if __name__ == "__main__":
    fig_isp_chain()
    fig_demosaic_compare()
    fig_zippering()
    fig_order_matters()
    fig_tone_and_detail()
