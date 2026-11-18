"""Check the test shots for capture-checklist steps 0.1 and 0.2 (EE 513 Homework 7 / Project 2).

Step 0.1, one DNG: is it native (Bayer) RAW, not ProRAW, and is it from the same camera as Project 1?
    python check_test_shots.py raw "Test/IMG_1234.DNG" --reference "../Project 1/Homework 2 Image Captures/Slanted Edge/IMG_0823.DNG"

Step 0.2, a RAW + JPEG/HEIC pair from one shutter press: are both there, from the same moment,
and does the DNG still pass step 0.1? Also saves a side-by-side PNG so you can see whether the
JPEG is Apple's normal processing or a Halide Look.
    python check_test_shots.py pair "Test/IMG_1235.DNG"                       # finds IMG_1235.JPG/.HEIC
    python check_test_shots.py pair "Test/IMG_1235.DNG" --jpeg "Test/IMG_1235.HEIC"

Needs numpy, rawpy, tifffile, Pillow and matplotlib (all in the course environment).
HEIC companions also need pillow-heif:  uv add pillow-heif  (or set Halide to save JPEG).
Exit code 0 = every check passed, 1 = something failed.
"""

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rawpy
import tifffile

SAT_DN = 3940.0          # Project 1 measured saturation (green sites pile up from ~3950 DN)
PHOTOMETRIC = {32803: "CFA (Bayer mosaic: native RAW)", 34892: "LinearRaw (demosaiced: ProRAW)",
               2: "RGB (preview)", 6: "YCbCr (preview)", 1: "grayscale (preview)"}
COMPANION_EXT = (".jpg", ".jpeg", ".heic", ".heif")


def ok(flag, text):
    print(f"  [{'PASS' if flag else 'FAIL'}] {text}")
    return bool(flag)


def note(text):
    print(f"  [info] {text}")


# --------------------------------------------------------------------------------------- metadata
def _tag(page, name):
    """A tag from the IFD itself or its EXIF sub-IFD, else None."""
    exif = page.tags["ExifTag"].value if "ExifTag" in page.tags else {}
    if name in exif:
        return exif[name]
    return page.tags[name].value if name in page.tags else None


def _all_ifds(tif):
    """Every IFD in the file, including SubIFDs (where DNGs keep the full-resolution raw image)."""
    stack = list(tif.pages)
    while stack:
        p = stack.pop(0)
        yield p
        subs = getattr(p, "pages", None)
        if subs:
            stack.extend(subs)


def _as_float(v):
    if v is None:
        return None
    if isinstance(v, (tuple, list)):
        if len(v) == 2 and all(isinstance(x, (int, np.integer)) for x in v):
            return v[0] / v[1] if v[1] else None
        return _as_float(v[0])
    return float(v)


def _capture_time(dt, subsec):
    """'YYYY:MM:DD HH:MM:SS' plus sub-seconds -> datetime, or None."""
    if not dt:
        return None
    try:
        t = datetime.strptime(str(dt).strip("\x00 ").strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    if subsec:
        s = str(subsec).strip("\x00 ").strip()
        if s.isdigit():
            t = t.replace(microsecond=int((s + "000000")[:6]))
    return t


def dng_info(path):
    with tifffile.TiffFile(path) as t:
        p0 = t.pages[0]
        layers = []
        for p in _all_ifds(t):
            photo = p.tags["PhotometricInterpretation"].value if "PhotometricInterpretation" in p.tags else None
            sub = p.tags["NewSubfileType"].value if "NewSubfileType" in p.tags else 0
            layers.append({"photometric": int(photo) if photo is not None else None,
                           "full_res": int(sub) == 0, "w": p.imagewidth, "h": p.imagelength})
        iso = _tag(p0, "ISOSpeedRatings") or _tag(p0, "PhotographicSensitivity")
        return {
            "make": _tag(p0, "Make"), "model": _tag(p0, "Model"), "lens": _tag(p0, "LensModel"),
            "software": _tag(p0, "Software"),
            "iso": int(iso[0] if isinstance(iso, (tuple, list)) else iso) if iso is not None else None,
            "exposure_s": _as_float(_tag(p0, "ExposureTime")),
            "time": _capture_time(_tag(p0, "DateTimeOriginal"), _tag(p0, "SubsecTimeOriginal")),
            "layers": layers,
        }


# ------------------------------------------------------------------------------------- step 0.1
def check_raw(path, reference=None, sat=SAT_DN):
    path = Path(path)
    print(f"\nStep 0.1: {path.name}")
    if not ok(path.is_file(), f"file exists ({path})"):
        return False
    info = dng_info(path)
    passed = True

    # 1. The full-resolution image in the DNG must be a CFA mosaic. ProRAW stores LinearRaw.
    full = [L for L in info["layers"] if L["full_res"] and L["photometric"] in (32803, 34892)]
    kinds = sorted({PHOTOMETRIC.get(L["photometric"], str(L["photometric"])) for L in full})
    passed &= ok(any(L["photometric"] == 32803 for L in full),
                 f"raw image is a Bayer mosaic; DNG says: {', '.join(kinds) or 'no raw image found'}")
    if any(L["photometric"] == 34892 for L in full):
        print("         -> This is ProRAW. In Halide, switch the RAW format to native RAW and retake.")

    # 2. What rawpy (and so your notebooks) will see.
    with rawpy.imread(str(path)) as r:
        pattern = None if r.raw_pattern is None else r.raw_pattern.copy()
        desc = r.color_desc.decode(errors="replace")
        is_flat = r.raw_type == rawpy.RawType.Flat
        raw = r.raw_image_visible
        black = [float(b) for b in r.black_level_per_channel]
        white = float(r.white_level)
        shape, vmax = raw.shape, int(raw.max())
        clip_frac = float(np.mean(raw >= sat))
    passed &= ok(is_flat and pattern is not None and pattern.shape == (2, 2),
                 f"rawpy sees one mosaic plane with a 2x2 pattern: {None if pattern is None else pattern.tolist()}")
    passed &= ok(desc == "RGBG", f"color_desc is 'RGBG' (got '{desc}')")
    if pattern is not None and pattern.shape == (2, 2):
        sites = {f"raw[{i}::2, {j}::2]": desc[pattern[i, j]] for i in range(2) for j in range(2)}
        greens = [k for k, v in sites.items() if v == "G"]
        passed &= ok(len(greens) == 2, f"two green sites per 2x2 block: {', '.join(greens)}")
        order = "".join(desc[pattern[i, j]] for i in range(2) for j in range(2))
        note(f"CFA layout {order} (reading order); first green = {greens[0] if greens else '-'}")
    note(f"mosaic {shape[1]} x {shape[0]} (W x H) -> half-size green image {shape[1] // 2} x {shape[0] // 2}")
    note(f"black level per channel {black}; WhiteLevel tag {white:.0f}; Project 1 saturation {sat:.0f} DN")
    note(f"largest raw value {vmax} DN; {clip_frac:.3%} of sites at or above {sat:.0f} DN"
         + (" (fine for a throwaway test, but keep bursts below this)" if clip_frac > 0 else ""))

    # 3. Same camera as Project 1.
    note(f"camera: {info['make']} {info['model']}; lens: {info['lens']}")
    note(f"ISO {info['iso']}; exposure {info['exposure_s']:.4g} s" if info["exposure_s"] else f"ISO {info['iso']}")
    if reference:
        reference = Path(reference)
        if ok(reference.is_file(), f"Project 1 reference exists ({reference.name})"):
            ref = dng_info(reference)
            passed &= ok(info["lens"] == ref["lens"] and info["model"] == ref["model"],
                         f"same camera and lens as Project 1 (reference: {ref['model']}; {ref['lens']})")
            with rawpy.imread(str(reference)) as rr:
                ref_black = [float(b) for b in rr.black_level_per_channel]
                ref_shape = rr.raw_image_visible.shape
                ref_pattern = rr.raw_pattern.tolist() if rr.raw_pattern is not None else None
            passed &= ok(ref_shape == shape and ref_pattern == (None if pattern is None else pattern.tolist()),
                         f"same mosaic size and CFA pattern as Project 1 ({ref_shape[1]} x {ref_shape[0]})")
            if ref_black != black:
                note(f"black level differs from the reference ({ref_black}); check it against Project 1")
        else:
            passed = False
    else:
        note("no --reference given: compare the lens above with your Project 1 DNG yourself")
    return passed


# ------------------------------------------------------------------------------------- step 0.2
def find_companion(dng):
    hits = [p for p in dng.parent.iterdir() if p.stem == dng.stem and p.suffix.lower() in COMPANION_EXT]
    return hits[0] if hits else None


def open_companion(path):
    from PIL import Image, ImageOps
    if path.suffix.lower() in (".heic", ".heif"):
        try:
            from pillow_heif import register_heif_opener
        except ImportError:
            raise SystemExit("  HEIC needs pillow-heif: run  uv add pillow-heif  (or set Halide to save JPEG).")
        register_heif_opener()
    img = Image.open(path)
    exif = img.getexif()
    sub = exif.get_ifd(0x8769)                      # EXIF sub-IFD
    meta = {"make": exif.get(271), "model": exif.get(272), "software": exif.get(305),
            "lens": sub.get(42036), "iso": sub.get(34855),
            "time": _capture_time(sub.get(36867), sub.get(37521)),
            "size": img.size}
    return ImageOps.exif_transpose(img).convert("RGB"), meta


def side_by_side(dng, img, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with rawpy.imread(str(dng)) as r:
        rgb = r.postprocess(use_camera_wb=True, half_size=True, output_bps=8)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, im, title in zip(axes, (rgb, np.asarray(img)),
                             ("DNG, rawpy default render\n(NOT Apple's processing; just to compare framing)",
                              "companion JPEG/HEIC\n(should look like a normal iPhone photo, not a Look)")):
        ax.imshow(im)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def check_pair(dng, jpeg=None, reference=None, sat=SAT_DN):
    dng = Path(dng)
    passed = check_raw(dng, reference, sat)
    print(f"\nStep 0.2: RAW + phone JPEG from one press")
    jpeg = Path(jpeg) if jpeg else (find_companion(dng) if dng.is_file() else None)
    if not ok(jpeg is not None and jpeg.is_file(),
              f"companion image found: {jpeg.name if jpeg else f'no {dng.stem}.JPG/.HEIC next to the DNG'}"):
        print("         -> Check Halide's capture format (RAW + JPEG/HEIC) and how you exported: "
              "AirDrop or Image Capture keeps both; some exports drop the JPEG.")
        return False
    img, jm = open_companion(jpeg)
    dm = dng_info(dng)

    # Same moment: both files carry the capture time of the one shutter press.
    if dm["time"] and jm["time"]:
        dt = abs((dm["time"] - jm["time"]).total_seconds())
        passed &= ok(dt < 1.0, f"same capture time (DNG {dm['time']:%H:%M:%S.%f}, "
                               f"{jpeg.suffix.upper()[1:]} {jm['time']:%H:%M:%S.%f}; {dt:.3f} s apart)")
    else:
        note("capture time missing from one file; can't confirm the same press from metadata")
    passed &= ok(jm["model"] in (None, dm["model"]), f"same phone ({jm['model']})")
    if jm["lens"] and dm["lens"]:
        passed &= ok(jm["lens"] == dm["lens"], f"same lens ({jm['lens']})")

    # Same framing: aspect ratios should match, whatever the orientation.
    with rawpy.imread(str(dng)) as r:
        h, w = r.sizes.height, r.sizes.width
    ar_d, ar_j = max(w, h) / min(w, h), max(img.size) / min(img.size)
    passed &= ok(abs(ar_d - ar_j) < 0.02, f"same aspect ratio (DNG {w} x {h}, companion {img.size[0]} x {img.size[1]})")

    # Which processing made the companion. Halide writes it into the file's XMP as halide:grade.
    m = re.search(rb"halide:grade(?:>|=\")([^<\"]+)", jpeg.read_bytes())
    grade = m.group(1).decode(errors="replace").strip() if m else None
    if grade is None:
        note("no halide:grade tag in the companion: if Halide made it, confirm by eye that it's Apple's processing")
    elif grade.endswith(".p0") or "zero" in grade.lower():
        passed &= ok(False, f"companion is Apple's processing (it is Halide's Process Zero: {grade})")
        print("         -> In Halide, change the Look from Process Zero to Apple's processing, retake, and rerun.")
    else:
        note(f"companion grade/Look: {grade}. Confirm this is Apple's processing (not a film Look) and "
             f"record it for the report.")
    note(f"companion Software tag: {jm['software']!r}; DNG Software tag: {dm['software']!r}")
    out_png = dng.with_name(f"{dng.stem}_pair_check.png")
    side_by_side(dng, img, out_png)
    note(f"saved {out_png.name}: open it and confirm the companion looks like a normal iPhone photo, "
         f"not a film Look. Record what it is for the report.")
    return passed


# --------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("raw", "pair"), help="raw = step 0.1, pair = step 0.2")
    ap.add_argument("dng", help="the test DNG")
    ap.add_argument("--jpeg", help="companion JPEG/HEIC (default: same name as the DNG)")
    ap.add_argument("--reference", help="a Project 1 DNG from the same camera, for the lens check")
    ap.add_argument("--sat", type=float, default=SAT_DN, help=f"saturation in DN (default {SAT_DN:.0f})")
    a = ap.parse_args()
    passed = check_raw(a.dng, a.reference, a.sat) if a.step == "raw" else check_pair(a.dng, a.jpeg, a.reference, a.sat)
    print("\nALL CHECKS PASSED" if passed else "\nSOME CHECKS FAILED (see [FAIL] above)")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()