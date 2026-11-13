"""Project 1: the two measurements the homework notebooks did not already produce.

1. Calibration uncertainty by leave-one-view-out jackknife, from the corners cached by Homework 2
   (hw2_checkerboard_corners.npz), for both the four- and five-coefficient distortion models,
   and a board-flatness check (homography fit to corners undistorted without their own view).
2. MTF50 on the corner squares of the slanted-edge target, in the same six captures and with the
   Homework 2 notebook's own functions (green_plane, slanted_edge_mtf, find_squares, edge_rois),
   which are read out of that notebook so there is one copy of the code.

Writes project1_checks.json and figures/p1_mtf50_by_field.png.
Run from the Project 1 folder:   python project1_checks.py
"""
import json
import re
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rawpy

BASE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else BASE
HW2_NB = BASE / "Measure Your Cameras Spatial Resolution.ipynb"
CORNERS = BASE / "hw2_checkerboard_corners.npz"
EDGE_DIR = BASE / "Homework 2 Image Captures" / "Slanted Edge"
PATTERN, SQUARE_MM = (9, 6), 20.0


# ----------------------------------------------------------------------------- 1. calibration
def calibrate(corners, names, image_size, k3):
    objp = np.zeros((PATTERN[0] * PATTERN[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM
    img = [corners[n] for n in names]
    rms, K, d, rv, tv, sd, _, _ = cv2.calibrateCameraExtended(
        [objp] * len(img), img, image_size, None, None, flags=0 if k3 else cv2.CALIB_FIX_K3)
    d, sd = d.ravel(), sd.ravel()
    vals = {"rms": rms, "f": 0.5 * (K[0, 0] + K[1, 1]), "fx": K[0, 0], "fy": K[1, 1],
            "cx": K[0, 2], "cy": K[1, 2], "k1": d[0], "k2": d[1], "p1": d[2], "p2": d[3], "k3": d[4]}
    fit_sd = {"f": 0.5 * np.hypot(sd[0], sd[1]), "fx": sd[0], "fy": sd[1], "cx": sd[2], "cy": sd[3],
              "k1": sd[4], "k2": sd[5], "p1": sd[6], "p2": sd[7], "k3": sd[8]}
    # depth of the board centre in each view (for the focus-breathing check)
    depth = [(cv2.Rodrigues(r)[0] @ np.array([80.0, 50.0, 0.0]) + t.ravel())[2] for r, t in zip(rv, tv)]
    return vals, fit_sd, np.array(depth)


def jackknife(corners, names, image_size, k3):
    full, fit_sd, depth = calibrate(corners, names, image_size, k3)
    reps = [calibrate(corners, [m for m in names if m != n], image_size, k3)[0] for n in names]
    n = len(names)
    se = {k: float(np.sqrt((n - 1) / n * sum((r[k] - np.mean([q[k] for q in reps])) ** 2 for r in reps)))
          for k in full}
    return {k: {"value": float(full[k]), "fit_sd": float(fit_sd.get(k, np.nan)), "jackknife_se": se[k]}
            for k in full}, depth


z = np.load(CORNERS, allow_pickle=True)
corners = z["corners"].item()
image_size = tuple(int(v) for v in z["image_size"])
names = sorted(corners)
calib = {}
for label, k3 in (("four_coef", False), ("five_coef", True)):
    calib[label], depth = jackknife(corners, names, image_size, k3)
calib["n_views"] = len(names)
calib["board_depth_mm"] = {"median": float(np.median(depth)), "min": float(depth.min()), "max": float(depth.max())}


# ----------------------------------------------------------------------------- 1b. board flatness
# A flat board maps to the image by a homography once lens distortion is removed. Fitting the
# homography to the RAW corners (the Homework 2 test) mixes distortion into the residual, so here
# each view is undistorted with a calibration that leaves that view out (no circularity), and the
# radial share of the raw residual shows how much of it is distortion. 5-coefficient model.
OBJ2 = (np.mgrid[0:PATTERN[0], 0:PATTERN[1]].T.reshape(-1, 2) * SQUARE_MM).astype(np.float32)


def homography_rms(pts):
    H, _ = cv2.findHomography(OBJ2, pts)
    res = pts - cv2.perspectiveTransform(OBJ2.reshape(-1, 1, 2), H).reshape(-1, 2)
    return float(np.sqrt((res ** 2).sum(axis=1).mean())), res


def k_and_dist(view_names):
    objp = np.c_[OBJ2, np.zeros(len(OBJ2), np.float32)]
    _, K, d, _, _ = cv2.calibrateCamera([objp] * len(view_names), [corners[n] for n in view_names],
                                        image_size, None, None)
    return K, d


K_all, _ = k_and_dist(names)
raw_rms, radial_share, loo_rms = [], [], []
for n in names:
    p = corners[n].reshape(-1, 2).astype(np.float64)
    e, res = homography_rms(p)
    raw_rms.append(e)
    u = p - K_all[:2, 2]
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    radial_share.append(float(((res * u).sum(axis=1) ** 2).sum() / (res ** 2).sum()))
    K2, d2 = k_and_dist([m for m in names if m != n])
    und = cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float32), K2, d2, P=K2).reshape(-1, 2)
    loo_rms.append(homography_rms(und.astype(np.float64))[0])

# Size of a bulge that would explain the whole undistorted residual: an out-of-plane shift dz at
# X mm off-axis, Z mm away, moves the image point by about f_px * X * dz / Z^2 (X = 100 mm here).
f_px = calib["five_coef"]["f"]["value"]
px_per_mm = f_px * 100.0 / calib["board_depth_mm"]["median"] ** 2
calib["flatness"] = {
    "raw_homography_rms_px": {"median": float(np.median(raw_rms)), "max": float(np.max(raw_rms))},
    "raw_residual_radial_share": float(np.median(radial_share)),
    "undistorted_leave_one_out_homography_rms_px": {"median": float(np.median(loo_rms)),
                                                     "max": float(np.max(loo_rms))},
    "image_shift_px_per_mm_out_of_plane": float(px_per_mm),
    "flatness_bound_mm": float(np.median(loo_rms) / px_per_mm),
}


# ----------------------------------------------------------------------------- 2. corner MTF
src = "\n".join("".join(c["source"]) for c in json.load(open(HW2_NB, encoding="utf-8"))["cells"]
                if c["cell_type"] == "code")
ns = {"np": np, "cv2": cv2, "rawpy": rawpy, "Path": Path, "pd": pd,
      "EDGE_HALF_WIDTH": 24, "MANUAL_CENTER": None}
for fn in ("green_plane", "slanted_edge_mtf", "find_squares", "edge_rois"):
    exec(re.search(rf"^def {fn}\(.*?(?=^\S)", src, re.S | re.M).group(0), ns)

rows = []
for f in sorted(EDGE_DIR.glob("*.DNG")):
    g, black, _, _ = ns["green_plane"](f)
    lin = g - black
    h, w = lin.shape
    half_diag = np.hypot(w / 2, h / 2)
    for sq in ns["find_squares"](lin):
        rel = sq["center"] - np.array([w / 2, h / 2])
        field = float(np.hypot(*rel) / half_diag)
        m = [ns["slanted_edge_mtf"](lin[r0:r1, c0:c1], f_max=1.5)["mtf50"]
             for _, r0, r1, c0, c1 in ns["edge_rois"](sq, lin.shape)]
        where = "centre" if field < 0.1 else ("top" if rel[1] < 0 else "bottom") + "-" + ("left" if rel[0] < 0 else "right")
        rows.append({"file": f.name, "square": where, "field": field, "mtf50": float(np.nanmean(m))})
sq = pd.DataFrame(rows)
centre = sq[sq.square == "centre"].set_index("file").mtf50
corner = sq[sq.square != "centre"]
per_cap = corner.groupby("file").mtf50.agg(["mean", "min"])
ratio = per_cap["mean"] / centre
by_sq = sq.groupby("square").agg(field=("field", "mean"), mean=("mtf50", "mean"),
                                 sd=("mtf50", lambda s: s.std(ddof=1)), n=("mtf50", "size"))
mtf = {
    "unit": "cycles per green-plane pixel (green-plane Nyquist = 0.5)",
    "n_captures": int(centre.size),
    "by_square": by_sq.round(4).reset_index().to_dict(orient="records"),
    "centre": [float(centre.mean()), float(centre.std(ddof=1))],
    "corner_squares_mean": [float(per_cap["mean"].mean()), float(per_cap["mean"].std(ddof=1))],
    "corner_squares_worst": [float(per_cap["min"].mean()), float(per_cap["min"].std(ddof=1))],
    "corner_over_centre": [float(ratio.mean()), float(ratio.std(ddof=1))],
    "corner_square_field": float(corner.field.mean()),
}

(OUT_DIR / "project1_checks.json").write_text(json.dumps({"calibration": calib, "mtf": mtf}, indent=2))

# ----------------------------------------------------------------------------- figure
order = ["centre", "top-left", "top-right", "bottom-right"]
b = by_sq.reindex(order)
fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
x = np.arange(len(order))
ax.bar(x, b["mean"], width=0.55, color="#2a78d6", edgecolor="#fcfcfb", linewidth=2)
ax.errorbar(x, b["mean"], yerr=b["sd"], fmt="none", ecolor="#0b0b0b", capsize=4, lw=1)
for xi, (m, s) in enumerate(zip(b["mean"], b["sd"])):
    ax.text(xi, m + s + 0.012, f"{m:.3f} ± {s:.3f}", ha="center", fontsize=8, color="#0b0b0b")
ax.set_xticks(x, [f"{s}\n({100 * f:.0f}% of half-diagonal)" for s, f in zip(order, b["field"])], fontsize=8)
ax.axhline(0.5, color="#52514e", lw=0.8, ls="--")
ax.text(len(order) - 0.55, 0.505, "green-plane Nyquist", fontsize=7, color="#52514e", ha="right", va="bottom")
ax.set_ylim(0, 0.56)
ax.set_ylabel("MTF50 (cycles / green px)", fontsize=8)
ax.set_title(f"MTF50 by position, mean ± s.d. over {centre.size} captures "
             "(bottom-left square not detected)", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8)
fig.tight_layout()
(OUT_DIR / "figures").mkdir(exist_ok=True)
fig.savefig(OUT_DIR / "figures" / "p1_mtf50_by_field.png", bbox_inches="tight")
print(json.dumps(mtf, indent=1))
print(json.dumps(calib["board_depth_mm"]))
