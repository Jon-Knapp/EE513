"""Project 1: the measurements the homework notebooks did not already produce.

Homework 2 session (Homework 2 Image Captures/, hw2_checkerboard_corners.npz)
1. Calibration uncertainty by leave-one-view-out jackknife, for the four- and five-coefficient
   models, and a board-flatness check (homography fit to corners undistorted without their own view).
2. MTF50 on the corner squares of the slanted-edge target (about 32% of the way to the corner),
   with the Homework 2 notebook's own functions, read out of that notebook so there is one copy.

Project 1 capture session (Project 1 Captures/)
3. Frame-corner MTF: the target placed so its outer squares reach 82-84% of the way to each frame
   corner. Uses a square detector that tolerates uneven lighting, validated first on the Homework 2
   captures against the original detector.
4. Ultra-wide (0.5x) calibration: five-coefficient and rational models, jackknife uncertainties,
   flatness check and coverage. Corners are detected with the Homework 2 detector and cached in
   uw_checkerboard_corners.npz.
5. Apple's own distortion model, read from the WarpRectilinear2 opcode in the ultra-wide DNG.
6. Straight-edge test: a metal ruler near the frame edge, one frame per camera, corrected with each
   calibration (and with Apple's model) and measured for straightness. Neither calibration saw it.

Writes project1_checks.json and figures/p1_mtf50_by_field.png, p1_mtf50_vs_field.png,
p1_distortion_and_ruler.png.
Run from the Project 1 folder:   python project1_checks.py
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rawpy

from camchar import (bow, bow_noise, dng_opcodes, f_apple, find_squares_multi, lens_id, median_se,
                     notebook_functions, radial_profile)

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
    # depth of the board center in each view (for the focus-breathing check)
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
                                                     "median_se": median_se(loo_rms),
                                                     "max": float(np.max(loo_rms))},
    "image_shift_px_per_mm_out_of_plane": float(px_per_mm),
    "flatness_bound_mm": float(np.median(loo_rms) / px_per_mm),
}


# ----------------------------------------------------------------------------- 2. corner MTF
ns = notebook_functions(HW2_NB, ("green_plane", "slanted_edge_mtf", "find_squares", "edge_rois"),
                        {"np": np, "cv2": cv2, "rawpy": rawpy, "Path": Path, "pd": pd,
                         "EDGE_HALF_WIDTH": 24, "MANUAL_CENTER": None})

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
        where = "center" if field < 0.1 else ("top" if rel[1] < 0 else "bottom") + "-" + ("left" if rel[0] < 0 else "right")
        rows.append({"file": f.name, "square": where, "field": field, "mtf50": float(np.nanmean(m))})
sq = pd.DataFrame(rows)
center = sq[sq.square == "center"].set_index("file").mtf50
corner = sq[sq.square != "center"]
per_cap = corner.groupby("file").mtf50.agg(["mean", "min"])
ratio = per_cap["mean"] / center
by_sq = sq.groupby("square").agg(field=("field", "mean"), mean=("mtf50", "mean"),
                                 sd=("mtf50", lambda s: s.std(ddof=1)), n=("mtf50", "size"))
mtf = {
    "unit": "cycles per green-plane pixel (green-plane Nyquist = 0.5)",
    "n_captures": int(center.size),
    "by_square": by_sq.round(4).reset_index().to_dict(orient="records"),
    "center": [float(center.mean()), float(center.std(ddof=1))],
    "corner_squares_mean": [float(per_cap["mean"].mean()), float(per_cap["mean"].std(ddof=1))],
    "corner_squares_worst": [float(per_cap["min"].mean()), float(per_cap["min"].std(ddof=1))],
    "corner_over_center": [float(ratio.mean()), float(ratio.std(ddof=1))],
    "corner_square_field": float(corner.field.mean()),
}

# ============================================================================= Project 1 captures
# Everything below uses the dedicated Project 1 capture session (folder "Project 1 Captures"):
#   Main Corner MTF/{center,top-left,top-right,bottom-left,bottom-right}  main camera, slanted edge
#   Ultra Wide Checkerboard/                                              0.5x camera, calibration
#   Straight Edge/                                                         one 0.5x and one 1x frame of a metal ruler
CAP = BASE / "Project 1 Captures"
MTF_DIR = CAP / "Main Corner MTF"
POSITIONS = ("center", "top-left", "top-right", "bottom-left", "bottom-right")   # folder names, as framed in Halide
ATTEMPT1_CENTER = [f"IMG_{i:04d}.DNG" for i in range(978, 983)]                  # first attempt (shot through a ring light)
UW_DIR = CAP / "Ultra Wide Checkerboard"
UW_CORNER_CACHE = BASE / "uw_checkerboard_corners.npz"
EDGE_FRAMES = {"ultra_wide": CAP / "Straight Edge" / "IMG_1125.DNG", "main": CAP / "Straight Edge" / "IMG_1126.DNG"}


# ----------------------------------------------------------------------------- 3. frame-corner MTF


def measure_squares(path, finder):
    """MTF50 (mean of the 4 edges) of every square found in one capture, with its field position."""
    g, black, _, _ = ns["green_plane"](path)
    lin = g - black
    h, w = lin.shape
    half_diag = np.hypot(w / 2, h / 2)
    rows = []
    for s in finder(lin):
        rel = s["center"] - np.array([w / 2, h / 2])
        m = [ns["slanted_edge_mtf"](lin[r0:r1, c0:c1], f_max=1.5)["mtf50"]
             for _, r0, r1, c0, c1 in ns["edge_rois"](s, lin.shape)]
        rows.append({"file": path.name, "x": float(s["center"][0]), "y": float(s["center"][1]),
                     "side": float(s["side"]), "field": float(np.hypot(*rel) / half_diag),
                     "n_edges": int(np.isfinite(m).sum()), "mtf50": float(np.nanmean(m))})
    return rows


# 3a. Validate the new detector on the Homework 2 captures, where the old one is known to work.
old = pd.DataFrame([r for f in sorted(EDGE_DIR.glob("*.DNG")) for r in measure_squares(f, ns["find_squares"])])
new = pd.DataFrame([r for f in sorted(EDGE_DIR.glob("*.DNG")) for r in measure_squares(f, find_squares_multi)])
merged = old.merge(new, on="file", suffixes=("_old", "_new"))
merged = merged[np.hypot(merged.x_old - merged.x_new, merged.y_old - merged.y_new) < 20]
detector_check = {
    "squares_found_old": int(len(old)), "squares_found_new": int(len(new)),
    "matched": int(len(merged)),
    "max_abs_mtf50_difference_on_matched_squares": float(np.abs(merged.mtf50_old - merged.mtf50_new).max()),
    "center_mtf50_old_new": [float(old[old.field < 0.1].mtf50.mean()), float(new[new.field < 0.1].mtf50.mean())],
}

# 3b. The frame-corner session
sq_rows = []
for pos in POSITIONS:
    for f in sorted((MTF_DIR / pos).glob("*.DNG")):
        for r in measure_squares(f, find_squares_multi):
            sq_rows.append({"position": pos, **r})
fc = pd.DataFrame(sq_rows)
fc["sx"], fc["sy"] = (fc.x / 25).round() * 25, (fc.y / 25).round() * 25           # same physical square across frames
per_square = (fc.groupby(["position", "sx", "sy"])
              .agg(field=("field", "mean"), n=("mtf50", "size"), mtf50=("mtf50", "mean"),
                   sd=("mtf50", lambda s: s.std(ddof=1))).reset_index())
center_sq = per_square[(per_square.position == "center")].sort_values("field").iloc[0]
outer = (per_square[per_square.position != "center"].sort_values("field")
         .groupby("position").tail(1).set_index("position").reindex(POSITIONS[1:]))
# Per-frame MTF50 of the outermost square of each corner folder, for the frame-to-frame spread
outer_frames = {p: fc[(fc.position == p) & (fc.sx == outer.loc[p, "sx"]) & (fc.sy == outer.loc[p, "sy"])].mtf50.to_numpy()
                for p in POSITIONS[1:]}

# Session-to-session (focus) uncertainty: the center value in three separate sessions
sessions = {"hw2_2026-11-11": float(center.mean()), "project1_attempt2": float(center_sq.mtf50)}
a1 = [MTF_DIR / n for n in ATTEMPT1_CENTER if (MTF_DIR / n).exists()]
if a1:
    a1_rows = pd.DataFrame([r for f in a1 for r in measure_squares(f, find_squares_multi)])
    sessions["project1_attempt1_ring_light"] = float(a1_rows[a1_rows.field < 0.1].mtf50.mean())
session_sd = float(np.std(list(sessions.values()), ddof=1))
rel_session = session_sd / float(center_sq.mtf50)

c_mean, c_worst = float(outer.mtf50.mean()), float(outer.mtf50.min())
frame_corner = {
    "unit": "cycles per green-plane pixel (green-plane Nyquist = 0.5)",
    "detector_check_on_hw2_captures": detector_check,
    "frames_per_position": {p: int(fc[fc.position == p].file.nunique()) for p in POSITIONS},
    "squares_measured": int(len(fc)),
    "center": {"field": float(center_sq.field), "mtf50": float(center_sq.mtf50), "frame_sd": float(center_sq.sd)},
    "outermost_square_per_corner": {p: {"field": float(outer.loc[p, "field"]), "mtf50": float(outer.loc[p, "mtf50"]),
                                        "frame_sd": float(np.std(outer_frames[p], ddof=1))} for p in POSITIONS[1:]},
    "corner_mean": c_mean, "corner_worst": c_worst, "worst_corner": str(outer.mtf50.idxmin()),
    "corner_over_center": c_mean / float(center_sq.mtf50),
    "session_center_values": sessions, "session_sd": session_sd,
    "results_json": {   # value, uncertainty: the session-to-session spread dominates, scaled to each value
        "mtf50_center_cy_per_px": [float(center_sq.mtf50), session_sd],
        "mtf50_corner_mean_cy_per_px": [c_mean, rel_session * c_mean],
        "mtf50_corner_worst_cy_per_px": [c_worst, rel_session * c_worst]},
    "all_squares": per_square.round(4).to_dict(orient="records"),
}


# ----------------------------------------------------------------------------- 4. ultra-wide calibration


uw_files = sorted(UW_DIR.glob("*.DNG"))
uw_ids = {lens_id(f) for f in uw_files}
assert len(uw_ids) == 1, f"more than one camera in the ultra-wide series: {uw_ids}"
uw_names = [f.name for f in uw_files]
cache_ok = False
if UW_CORNER_CACHE.exists():
    zc = np.load(UW_CORNER_CACHE, allow_pickle=True)
    cache_ok = list(zc["names"]) == uw_names
if cache_ok:
    uw_corners = zc["corners"].item(); uw_size = tuple(int(v) for v in zc["image_size"])
else:
    det = {"np": np, "cv2": cv2, "rawpy": rawpy, "Path": Path, "PATTERN": PATTERN, "SCALE": 4,
           "criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)}
    notebook_functions(HW2_NB, ("load_rgb", "detect_corners"), det)   # the Homework 2 detector, unchanged
    uw_corners = {}
    for f in uw_files:
        c, uw_size = det["detect_corners"](f)
        if c is not None:
            uw_corners[f.name] = c
    np.savez(UW_CORNER_CACHE, names=np.array(uw_names), corners=np.array(uw_corners, dtype=object),
             image_size=np.array(uw_size), pattern=np.array(PATTERN))
uw_used = sorted(uw_corners)

MODELS = {"five_coef": 0, "rational": cv2.CALIB_RATIONAL_MODEL}
DIST_NAMES = {"five_coef": ["k1", "k2", "p1", "p2", "k3"],
              "rational": ["k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"]}


def calibrate_model(cmap, view_names, size, flags):
    objp = np.c_[OBJ2, np.zeros(len(OBJ2), np.float32)]
    rms, K, d, _, _ = cv2.calibrateCamera([objp] * len(view_names), [cmap[n] for n in view_names], size, None, None, flags=flags)
    return rms, K, d.ravel()


def jackknife_model(cmap, view_names, size, model, fracs=(), corner_r=1.0):
    def vec(rms, K, d):
        return np.r_[rms, 0.5 * (K[0, 0] + K[1, 1]), K[0, 2], K[1, 2], d[:len(DIST_NAMES[model])]]
    def prof(rms, K, d):
        return radial_profile(K, d, fracs, corner_r)
    fits = [calibrate_model(cmap, view_names, size, MODELS[model])] + \
           [calibrate_model(cmap, [m for m in view_names if m != n], size, MODELS[model]) for n in view_names]
    full, reps = vec(*fits[0]), np.array([vec(*f_) for f_ in fits[1:]])
    n = len(view_names)
    jse = lambda r: np.sqrt((n - 1) / n * ((r - r.mean(0)) ** 2).sum(0))
    se = jse(reps)
    labels = ["rms", "f", "cx", "cy"] + DIST_NAMES[model]
    out = {k: {"value": float(v), "jackknife_se": float(s)} for k, v, s in zip(labels, full, se)}
    if len(fracs):
        p_reps = np.array([prof(*f_) for f_ in fits[1:]])
        out["radial_distortion_pct_jackknife_se_at"] = dict(zip(map(str, fracs), map(float, jse(p_reps))))
    return out


def loo_flatness(cmap, view_names, size, flags):
    res = []
    for n in view_names:
        _, K2, d2 = calibrate_model(cmap, [m for m in view_names if m != n], size, flags)
        u = cv2.undistortPoints(cmap[n].reshape(-1, 1, 2).astype(np.float32), K2, d2, P=K2).reshape(-1, 2)
        res.append(homography_rms(u.astype(np.float64))[0])
    return {"median": float(np.median(res)), "median_se": median_se(res), "max": float(np.max(res))}


def coverage(cmap, view_names, K, size):
    pts = np.concatenate([cmap[n].reshape(-1, 2) for n in view_names])
    nr = lambda p: np.hypot((p[:, 0] - K[0, 2]) / K[0, 0], (p[:, 1] - K[1, 2]) / K[1, 1])
    fc_ = np.array([[0, 0], [size[0] - 1, 0], [0, size[1] - 1], [size[0] - 1, size[1] - 1]], float)
    return float(nr(pts).max() / nr(fc_).max())


corner_r = float(np.hypot(image_size[0] / 2, image_size[1] / 2))
fracs = [0.25, 0.5, 0.75, 0.83]
uw = {"lens_id": list(next(iter(uw_ids))), "n_views": len(uw_used), "n_files": len(uw_files)}
for model in MODELS:
    uw[model] = jackknife_model(uw_corners, uw_used, uw_size, model, fracs, corner_r)
    _, K_m, d_m = calibrate_model(uw_corners, uw_used, uw_size, MODELS[model])
    uw[model]["flatness_loo_px"] = loo_flatness(uw_corners, uw_used, uw_size, MODELS[model])
    uw[model]["radial_distortion_pct_at"] = dict(zip(map(str, fracs), radial_profile(K_m, d_m, fracs, corner_r)))
_, K_uw, d_uw = calibrate_model(uw_corners, uw_used, uw_size, MODELS["rational"])
uw["coverage_of_corner_radius"] = coverage(uw_corners, uw_used, K_uw, uw_size)
uw["focal_length_mm_exif"] = uw["lens_id"][1]

# Main camera on the same footing (adopted 5-coefficient model, HW2 corners) for the comparison
_, K_main, d_main = calibrate_model(corners, names, image_size, 0)
main_profile = {"coverage_of_corner_radius": coverage(corners, names, K_main, image_size),
                "radial_distortion_pct_at": dict(zip(map(str, fracs), radial_profile(K_main, d_main, fracs, corner_r))),
                "radial_distortion_pct_jackknife_se_at":
                    jackknife_model(corners, names, image_size, "five_coef", fracs, corner_r)["radial_distortion_pct_jackknife_se_at"]}


# ----------------------------------------------------------------------------- 5. Apple's own distortion model


apple = {"ultra_wide_opcodes": [o["opcode"] for o in dng_opcodes(uw_files[0])],
         "main_opcodes": [o["opcode"] for o in dng_opcodes(sorted((MTF_DIR / "center").glob("*.DNG"))[0])]}
RW = np.hypot(uw_size[0] / 2, uw_size[1] / 2)               # the opcode's radius unit: center to frame corner


def apple_coeffs(path):
    w = next((o for o in dng_opcodes(path) if "radial_coefficients" in o), None)
    return None if w is None else np.array(w["radial_coefficients"])


warps = {f.name: apple_coeffs(f) for f in uw_files}
warps = {k: v for k, v in warps.items() if v is not None}
warp = bool(warps)
if warp:
    K_all_apple = np.array(list(warps.values()))
    # Apple's predicted raw displacement (%) at the same radii as mine, in every file: it changes per capture
    prof = np.array([[100 * (1 / f_apple(fr_, kr) - 1) for fr_ in fracs] for kr in K_all_apple])
    apple["files_with_warp"] = len(warps)
    apple["radial_displacement_pct_at"] = {str(fr_): {"mean": float(prof[:, i].mean()), "sd": float(prof[:, i].std(ddof=1)),
                                                      "min": float(prof[:, i].min()), "max": float(prof[:, i].max())}
                                           for i, fr_ in enumerate(fracs)}
    # Agreement with my rational model over the calibrated range, per file, with a free overall scale
    r_raw = np.linspace(20, uw["coverage_of_corner_radius"] * RW, 400)
    pts = np.c_[K_uw[0, 2] + r_raw, np.full_like(r_raw, K_uw[1, 2])].reshape(-1, 1, 2).astype(np.float32)
    mine = cv2.undistortPoints(pts, K_uw, d_uw, P=K_uw).reshape(-1, 2)[:, 0] - K_uw[0, 2]
    dmax, drms, scale, dmax_noscale = [], [], [], []
    for kr in K_all_apple:
        a_ = r_raw * f_apple(r_raw / RW, kr)
        dd = (a_ @ mine) / (a_ @ a_) * a_ - mine
        dmax.append(np.abs(dd).max()); drms.append(np.sqrt(np.mean(dd ** 2)))
        scale.append(float((a_ @ mine) / (a_ @ a_))); dmax_noscale.append(float(np.abs(a_ - mine).max()))
    apple["agreement_with_rational_px"] = {"max_over_files": float(np.max(dmax)), "median_of_max": float(np.median(dmax)),
                                          "median_rms": float(np.median(drms)),
                                          "fitted_scale_min_max": [min(scale), max(scale)],
                                          "without_scale_max_px": {"median": float(np.median(dmax_noscale)),
                                                                   "min": min(dmax_noscale), "max": max(dmax_noscale)}}
    kr_mean = K_all_apple.mean(0)


# ----------------------------------------------------------------------------- 6. straight-edge test
def ruler_edge(path):
    """Sub-pixel y of the metal ruler's top edge (paper above, metal below) in every 4th column."""
    with rawpy.imread(str(path)) as r:
        g = cv2.cvtColor(r.postprocess(use_camera_wb=True, user_flip=0, output_bps=16), cv2.COLOR_RGB2GRAY).astype(np.float64)
    dy = np.diff(cv2.GaussianBlur(g, (0, 0), 2), axis=0); H, W = g.shape; b0, b1 = int(0.80 * H), int(0.98 * H)
    thr = -0.25 * np.abs(dy[b0:b1, :]).max(); xs, ys = [], []
    for x in range(40, W - 40, 4):
        col = dy[b0:b1, x]; i = int(np.argmin(col))
        if col[i] < thr and 1 <= i < col.size - 1:
            a, bb, c = col[i - 1], col[i], col[i + 1]
            xs.append(x); ys.append(b0 + i + 0.5 + 0.5 * (a - c) / (a - 2 * bb + c))
    xs, ys = np.array(xs, float), np.array(ys, float); keep = np.ones_like(xs, bool)
    for _ in range(4):                                     # a smooth curve keeps the bent ends, drops the ruler's markings
        p = np.polyfit(xs[keep], ys[keep], 6); keep = np.abs(ys - np.polyval(p, xs)) < 3.0
    return xs[keep], ys[keep]


edge = {}
cams = {"ultra_wide": (uw_corners, uw_used, uw_size, ("five_coef", "rational")),
        "main": (corners, names, image_size, ("five_coef",))}
edge_curves = {}
for cam, path in EDGE_FRAMES.items():
    assert lens_id(path)[1] == (uw["lens_id"][1] if cam == "ultra_wide" else 6.765), f"{path.name} is not the {cam} camera"
    xs, ys = ruler_edge(path)
    cmap, vn, size, models = cams[cam]
    e = {"file": path.name, "n_points": int(len(xs)), "raw_pv_px": bow(xs, ys)[0], "raw_noise": bow_noise(xs, ys)}
    edge_curves[cam] = {"x": xs, "raw": bow(xs, ys)[1]}
    for model in models:
        _, K_m, d_m = calibrate_model(cmap, vn, size, MODELS[model])
        rn = np.hypot((xs - K_m[0, 2]) / K_m[0, 0], (ys - K_m[1, 2]) / K_m[1, 1])
        cov = coverage(cmap, vn, K_m, size) * np.hypot((size[0] / 2) / K_m[0, 0], (size[1] / 2) / K_m[1, 1])
        u = cv2.undistortPoints(np.c_[xs, ys].reshape(-1, 1, 2).astype(np.float32), K_m, d_m, P=K_m).reshape(-1, 2).astype(np.float64)
        inside = rn <= cov
        e[model] = {"corrected_pv_px": bow(u[:, 0], u[:, 1])[0],
                    "inside_calibrated_fraction": float(inside.mean()),
                    "inside_raw_pv_px": bow(xs[inside], ys[inside])[0],
                    "inside_corrected_pv_px": bow(u[inside, 0], u[inside, 1])[0],
                    "noise": bow_noise(u[:, 0], u[:, 1]), "inside_noise": bow_noise(u[inside, 0], u[inside, 1])}
        edge_curves[cam][model] = bow(u[:, 0], u[:, 1])[1]
    if cam == "ultra_wide" and warp:
        kr_frame = apple_coeffs(path)                                 # this frame's own coefficients
        cxy = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2]); v = np.c_[xs, ys] - cxy
        ca = cxy + v * f_apple(np.hypot(*v.T) / RW, kr_frame)[:, None]  # Apple's warp applied raw -> corrected
        e["apple_corrected_pv_px"] = bow(ca[:, 0], ca[:, 1])[0]
        e["apple_noise"] = bow_noise(ca[:, 0], ca[:, 1])
        edge_curves[cam]["apple"] = bow(ca[:, 0], ca[:, 1])[1]
    edge[cam] = e

(OUT_DIR / "project1_checks.json").write_text(json.dumps(
    {"calibration": calib, "mtf": mtf, "frame_corner_mtf": frame_corner,
     "ultra_wide_calibration": uw, "main_distortion_profile": main_profile,
     "apple_dng_opcodes": apple, "straight_edge": edge}, indent=2))

# ----------------------------------------------------------------------------- figures for the report
INK, MUTED, C1, C2, C3 = "#0b0b0b", "#52514e", "#2a78d6", "#eb6834", "#1baf7a"
(OUT_DIR / "figures").mkdir(exist_ok=True)

# MTF50 against position: this session, every square; Homework 2 session for comparison
fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=200)
ax.plot(sq.field, sq.mtf50, "o", ms=4, mfc="none", color=MUTED, label="HW2 session (6 frames)")
ax.plot(per_square.field, per_square.mtf50, "o", ms=4.5, color=C1, label="Project 1 session (5 frames each)")
ax.errorbar(per_square.field, per_square.mtf50, yerr=per_square.sd, fmt="none", ecolor=C1, lw=0.8)
for p in POSITIONS[1:]:
    ax.annotate(p, (outer.loc[p, "field"], outer.loc[p, "mtf50"]), xytext=(5, 0), textcoords="offset points",
                ha="left", va="center", fontsize=6, color=INK)
ax.set_xlim(0, 1.08); ax.set_ylim(0, 0.55)
ax.axhline(0.5, color=MUTED, lw=0.6, ls="--")
ax.set_xlabel("distance from center (fraction of half-diagonal)", fontsize=7)
ax.set_ylabel("MTF50 (cycles / green px)", fontsize=7)
ax.tick_params(labelsize=6.5); ax.legend(fontsize=6, frameon=False, loc="lower left")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT_DIR / "figures" / "p1_mtf50_vs_field.png", bbox_inches="tight"); plt.close(fig)

# Distortion profiles and the straight-edge test
fig, (a1x, a2x) = plt.subplots(1, 2, figsize=(7.2, 2.5), dpi=200)
fr = np.linspace(0.02, 0.98, 120)
for lab, (K_m, d_m, cov, col) in {"main (5-coef)": (K_main, d_main, main_profile["coverage_of_corner_radius"], C1),
                                   "ultra-wide (rational)": (K_uw, d_uw, uw["coverage_of_corner_radius"], C2)}.items():
    prof = np.array(radial_profile(K_m, d_m, fr, corner_r))
    a1x.plot(fr[fr <= cov], prof[fr <= cov], "-", color=col, lw=1.6, label=lab)
    ext = (fr > cov) & (fr <= 0.90)                       # a rational model diverges soon after its data ends
    a1x.plot(fr[ext], prof[ext], ":", color=col, lw=1.2)
if warp:
    a1x.plot(fr, 100 * (1 / f_apple(fr, kr_mean) - 1), "--", color=INK, lw=0.9, label="Apple DNG warp (ultra-wide, mean)")
a1x.axhline(0, color=MUTED, lw=0.5)
a1x.set_ylim(-6, 6); a1x.set_xlim(0, 1)
a1x.set_xlabel("distance from center (fraction of half-diagonal)", fontsize=7)
a1x.set_ylabel("radial displacement in raw (%)", fontsize=7)
a1x.set_title("(a) Distortion; dotted = beyond the checkerboard's reach", fontsize=7)
a1x.legend(fontsize=6, frameon=False); a1x.tick_params(labelsize=6.5)
xu = edge_curves["ultra_wide"]["x"]
a2x.plot(xu, edge_curves["ultra_wide"]["raw"], "-", color=MUTED, lw=1.2, label=f"raw ({edge['ultra_wide']['raw_pv_px']:.0f} px p-v)")
a2x.plot(xu, edge_curves["ultra_wide"]["rational"], "-", color=C2, lw=1.4,
         label=f"my rational model ({edge['ultra_wide']['rational']['corrected_pv_px']:.0f} px)")
if warp:
    a2x.plot(xu, edge_curves["ultra_wide"]["apple"], "--", color=INK, lw=0.9,
             label=f"Apple warp ({edge['ultra_wide']['apple_corrected_pv_px']:.0f} px)")
a2x.set_xlabel("x along the ruler (px)", fontsize=7); a2x.set_ylabel("deviation from a straight line (px)", fontsize=7)
a2x.set_title("(b) Ultra-wide: metal ruler near the bottom edge", fontsize=7)
a2x.legend(fontsize=6, frameon=False, loc="lower center"); a2x.tick_params(labelsize=6.5)
for ax_ in (a1x, a2x):
    for s in ("top", "right"):
        ax_.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT_DIR / "figures" / "p1_distortion_and_ruler.png", bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------------------------- figure (Homework 2 session corner squares)
order = ["center", "top-left", "top-right", "bottom-right"]
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
ax.set_title(f"MTF50 by position, mean ± s.d. over {center.size} captures "
             "(bottom-left square not detected)", fontsize=9)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.tick_params(labelsize=8)
fig.tight_layout()
(OUT_DIR / "figures").mkdir(exist_ok=True)
fig.savefig(OUT_DIR / "figures" / "p1_mtf50_by_field.png", bbox_inches="tight")
print(json.dumps(mtf, indent=1))
print(json.dumps(calib["board_depth_mm"]))
