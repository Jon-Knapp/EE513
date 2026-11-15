"""camchar: the reusable pieces of the Project 1 camera characterization.

Everything here is a plain function of arrays or of one file, so it can be tested without the
captures (see tests/). project1_checks.py imports these; the homework notebooks keep their own
functions, and notebook_functions() loads those by name so there is still only one copy of each.

    notebook_functions  load named functions out of a homework notebook
    find_squares_local  dark-square finder that tolerates uneven light (frame-corner MTF)
    find_squares_multi  union of two window sizes of find_squares_local
    lens_id             the assignment's same-camera check for a DNG
    parse_opcode_list   DNG opcode-list parser (WarpRectilinear2 coefficients)
    dng_opcodes         the opcode list stored in a DNG's raw IFD
    f_apple             radial factor of a WarpRectilinear2 opcode
    radial_profile      radial displacement (%) of an OpenCV distortion model
    bow, bow_noise      straightness of an edge (peak-to-valley about a line) and its noise
    median_se           bootstrap standard error of a median
"""
import json
import re
import struct

import cv2
import numpy as np
import tifffile
from scipy.ndimage import uniform_filter1d

OPNAMES = {1: "WarpRectilinear", 3: "FixVignetteRadial", 14: "WarpRectilinear2"}


def notebook_functions(nb_path, names, namespace=None):
    """Execute the top-level `def` of each named function from a notebook's code cells into
    `namespace` (a dict holding whatever globals those functions need) and return it."""
    ns = {} if namespace is None else namespace
    src = "\n".join("".join(c["source"]) for c in json.load(open(nb_path, encoding="utf-8"))["cells"]
                    if c["cell_type"] == "code")
    for fn in names:
        m = re.search(rf"^def {fn}\(.*?(?=^\S)", src + "\n#", re.S | re.M)
        if m is None:
            raise KeyError(f"{fn} is not defined in {nb_path}")
        exec(m.group(0), ns)
    return ns


def find_squares_local(lin, reduce=4, win_frac=0.12, thresh=0.5):
    """Dark square blobs on white paper, robust to uneven light. The Homework 2 find_squares uses one
    global Otsu threshold, which fails when a lamp's hot spot leaves the paper near the frame edge at
    ~15% of the center's brightness: the dim paper falls below the threshold and merges with the
    squares. Here each pixel is compared with the local paper level (the brightest value within a
    window larger than a square), with the same shape tests as find_squares."""
    h, w = lin.shape
    small = cv2.resize(lin.astype(np.float32), (w // reduce, h // reduce), interpolation=cv2.INTER_AREA)
    k = int(win_frac * max(small.shape)) | 1
    paper = cv2.GaussianBlur(cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))), (0, 0), k / 4)
    bw = ((small / np.maximum(paper, 1e-6) < thresh) & (paper > 0.05 * np.percentile(paper, 99))).astype(np.uint8) * 255
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = []
    for c in cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]:
        area = cv2.contourArea(c)
        if not (2e-4 * small.size < area < 0.25 * small.size):
            continue
        rect = cv2.minAreaRect(c); rw, rh = rect[1]
        if min(rw, rh) / max(rw, rh) < 0.8 or area / (rw * rh) < 0.85:
            continue
        box = cv2.boxPoints(rect) * reduce + (reduce - 1) / 2
        if box[:, 0].min() < 2 or box[:, 1].min() < 2 or box[:, 0].max() > w - 3 or box[:, 1].max() > h - 3:
            continue                                                       # cut by the frame edge
        out.append({"center": np.array(rect[0]) * reduce + (reduce - 1) / 2, "box": box, "side": 0.5 * (rw + rh) * reduce})
    return out


def find_squares_multi(lin):
    """Union over two window sizes (one smaller than the big square, one larger), de-duplicated."""
    out = []
    for fr in (0.12, 0.25):
        for s in find_squares_local(lin, win_frac=fr):
            if all(np.hypot(*(s["center"] - o["center"])) > 0.5 * min(s["side"], o["side"]) for o in out):
                out.append(s)
    return out


def lens_id(path):
    """The assignment's same-camera check: (lens model, focal length mm, 35 mm-equivalent, frame size)."""
    with tifffile.TiffFile(path) as t:
        p = t.pages[0]; exif = p.tags["ExifTag"].value; fl = exif["FocalLength"]
        return (exif.get("LensModel", "unknown"), round(fl[0] / fl[1], 3), exif.get("FocalLengthIn35mmFilm"),
                (p.tags["ImageWidth"].value, p.tags["ImageLength"].value))


def parse_opcode_list(b):
    """Parse a DNG opcode list (big-endian bytes, as stored in OpcodeList1/2/3). Returns one dict per
    opcode with its name, and for WarpRectilinear2 (id 14) its 15 radial coefficients kr0..kr14."""
    n, off, out = struct.unpack(">I", b[:4])[0], 4, []
    for _ in range(n):
        oid, _, _, size = struct.unpack(">IIII", b[off:off + 16]); data = b[off + 16:off + 16 + size]
        item = {"opcode": OPNAMES.get(oid, oid)}
        if oid == 14:                                     # planes, then 15 radial coefficients kr0..kr14
            item["radial_coefficients"] = list(struct.unpack(">15d", data[4:4 + 120]))
        out.append(item); off += 16 + size
    return out


def dng_opcodes(path):
    """Opcodes in OpcodeList3 (tag 51022) of the raw IFD of a DNG."""
    with tifffile.TiffFile(path) as t:
        pages = [p for p in t.pages[0].pages if 51022 in p.tags]
        if not pages:
            return []
        b = pages[0].tags[51022].value
    return parse_opcode_list(b)


def f_apple(r, kr):
    """Apple's radial factor; applied raw -> corrected (r_corrected = r_raw * f), which is the direction
    that straightens the metal ruler. r is normalized to the center-to-corner distance."""
    return np.polyval(kr[::-1], r)


def radial_profile(K, d, fracs, corner_r):
    """Radial displacement (%) of the distortion model at given fractions of the corner radius (raw radius)."""
    out = []
    for fr in fracs:
        pts = np.array([[[K[0, 2] + fr * corner_r, K[1, 2]]]], np.float32)
        ideal = cv2.undistortPoints(pts, K, d, P=K).reshape(2)[0] - K[0, 2]
        out.append(float(100 * (fr * corner_r / ideal - 1)))
    return out


def median_se(values, n_boot=2000, seed=0):
    """Bootstrap standard error of the median of per-view values."""
    v = np.asarray(values, float); rng = np.random.default_rng(seed)
    return float(np.std([np.median(rng.choice(v, v.size)) for _ in range(n_boot)], ddof=1))


def bow(x, y):
    p = np.polyfit(x, y, 1); r = y - np.polyval(p, x)
    return float(np.ptp(r)), r


def bow_noise(x, y, window=11):
    """Single-frame uncertainty of bow(): rms scatter of the edge points about their running mean
    (11 points, about 44 px of ruler), and how much the peak-to-valley changes when it is measured on
    that running mean instead of on the individual points (the part of bow() that is edge noise)."""
    pv, r = bow(x, y)
    smooth = uniform_filter1d(r, window, mode="nearest")
    return {"edge_noise_rms_px": float(np.std(r - smooth, ddof=1)), "smooth_pv_px": float(np.ptp(smooth)),
            "pv_change_px": float(abs(pv - np.ptp(smooth)))}

