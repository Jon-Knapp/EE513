# Characterizing the iPhone 16 Pro Max camera from raw files

EE 513 (Intro to Image Processing), Portland State University, Fall 2026. Project 1: *Characterize Your Camera*.

**The problem.** A phone's spec sheet lists megapixels and an f-number, and almost nothing that
decides image quality. This project measures the iPhone 16 Pro Max main (1×) camera the way an
image-quality engineer would, from Bayer raw DNGs, and publishes the result as a datasheet in which
every number carries an uncertainty and a stated method. As an extension it repeats the geometry
for the 0.5× ultra-wide camera and checks both calibrations against a straight metal ruler and
against Apple's own distortion correction stored in the DNG.

The report is [`Project 1/Project 1 Report IEEE.ipynb`](Project%201/Project%201%20Report%20IEEE.ipynb)
(a PDF preview sits beside it). Machine-readable results are in
[`Project 1/results.json`](Project%201/results.json) (main camera, course template) and
[`Project 1/results_ultrawide.json`](Project%201/results_ultrawide.json).

## Results (main camera unless noted)

| Quantity | Value | Uncertainty from |
|---|---|---|
| Focal length | 2901.8 ± 4.6 px (7.03 ± 0.15 mm) | leave-one-view-out jackknife, 25 views; mm bound from the 24 mm-equivalent rounding |
| Principal point (x, y) | 2026.1 ± 4.3, 1510.0 ± 4.4 px | jackknife |
| Radial k1, k2, k3 | 0.232 ± 0.008, −0.946 ± 0.038, 1.070 ± 0.063 | jackknife |
| Tangential p1, p2 | (1.02 ± 0.51, −0.75 ± 0.62) × 10⁻³ | jackknife |
| RMS reprojection error | 0.863 ± 0.029 px | jackknife |
| MTF50, center | 0.453 ± 0.064 cycles per green-plane pixel (0.91 of Nyquist) | s.d. over 3 capture sessions |
| MTF50, frame corners (82–84% of the way out) | 0.395, 0.374, 0.344, 0.301 (each ± 14%) | session spread, scaled |
| MTF50, corner mean | 0.353 ± 0.050 (78% of the center) | session spread, scaled |
| Conversion gain (ISO 50) | 0.246 ± 0.021 DN/e⁻ | difference-of-pairs PTC fit ± fit-range systematic |
| Read noise | 6.9 ± 0.6 e⁻ (1.694 ± 0.002 DN) | dark-frame pairs |
| Full well (clip level) | 14,500 ± 1,200 e⁻ | 3567 DN / gain |
| Dark current | < 0.6 e⁻/s (95% bound) | 23 lens-capped frames, 1/30 to 1 s |
| Dynamic range | 11.04 ± 0.06 stops | full scale / read noise; bound from the effective white level |
| Mean ΔE00, before → after color matrix | 12.35 ± 0.08 → 3.06 ± 0.04 | s.d. over 3 ColorChecker charts |
| Worst patch, before → after | 23.20 ± 0.47 → 8.89 ± 0.26 (after: patch 15, red) | s.d. over 3 charts |
| **Ultra-wide** focal length | 1620.4 ± 1.9 px (2.21 ± 0.08 mm) | jackknife, 33 views, rational model |
| **Ultra-wide** raw distortion at 50% / 83% of the corner radius | −0.56 ± 0.04% / −1.52 ± 0.07% | jackknife |
| **Ultra-wide** agreement with Apple's `WarpRectilinear2` correction | within 6.0 px in all 33 files | after one free scale factor (9.6 px worst without it) |

## Method in one paragraph per measurement

- **Geometry.** `cv2.calibrateCamera` on 25 checkerboard views (9 × 6, 20 mm squares), five
  distortion coefficients; uncertainties by leave-one-view-out jackknife, because the fit's own
  covariance understates them 3–6×. A homography on each view's undistorted corners (that view left
  out of the calibration) shows the board is flat to about 0.2 mm.
- **MTF.** Slanted edge (after ISO 12233) on one black-subtracted green plane of the Bayer mosaic,
  so no demosaicing enters the edge. Frame-corner squares are found with a local-threshold finder
  that tolerates uneven light. The uncertainty is the spread between capture sessions (focus), not
  within one.
- **Noise.** Photon transfer from flat-field pairs (½ Var(A − B) cancels fixed-pattern PRNU), a
  1/variance-weighted fit of Var = Kμ + K²σr² with the 1/12 DN² rounding variance removed, read
  noise from dark-frame pairs, and a comparison with the DNG's own `NoiseProfile`.
- **Color.** Quarter-resolution RGB without demosaicing, flat-field corrected, white balance from
  patch 22, a least-squares 3 × 3 matrix to XYZ, CIEDE2000 against colour-science's
  *ColorChecker24 – After November 2014* reference.
- **Beyond the spec.** The ultra-wide calibrated with the same code (five-coefficient and rational
  models), Apple's distortion polynomial read out of the DNG opcode list, and a metal ruler near the
  frame edge as a straight line neither calibration was fitted to.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/) and the raw captures (next section). From the repository root:

```bash
uv sync
uv run python "Project 1/reproduce.py"
```

`reproduce.py` checks that the captures are in place, runs `project1_checks.py` (jackknife
calibration, frame-corner MTF, ultra-wide calibration, Apple's opcode, metal-ruler test), runs the
test suite, checks every `results.json` entry against the template and against the values just
computed, and re-executes the report notebook and exports it to HTML. Options:

- `--all` re-runs the three homework notebooks first (geometry and MTF, noise, color), so every
  entry of `results.json` is recomputed from raw data. It needs `homework-captures.zip` as well
  (next section) and takes about 4 minutes.
- `--pdf` also prints the report to PDF with headless Chrome, if it is on the PATH.

The noise and dynamic-range entries of `results.json` are transcribed from the Homework 3
notebook's printed output; `reproduce.py --all` re-runs that notebook and reproduces its output
exactly, and the check in step 5 covers the geometry, MTF and color entries.

## Raw data

The DNGs are not in git (`*.DNG` is in `.gitignore`). They are two files on this repository's
[Releases](https://github.com/Jon-Knapp/EE513/releases) page:

- `Project1-captures.zip` (1.3 GB, 91 DNGs): everything the default run needs.
- `homework-captures.zip` (1.0 GB, 81 DNGs): the Homework 2 checkerboard, Homework 3 flat and dark
  frames, and Homework 4 color charts, needed only for `--all`.

Unzip both into `Project 1/` (on Windows, remove the extra folder name that "Extract All" suggests),
so that these folders exist:

```
Project 1/Homework 2 Image Captures/Slanted Edge/        6 slanted-edge frames (Homework 2 session)
Project 1/Project 1 Captures/Main Corner MTF/            center, top-left, top-right, bottom-left, bottom-right
                                                         (5 frames each), plus 25 frames from the failed
                                                         ring-light session in the folder itself;
                                                         IMG_0978-0982 give its center value
Project 1/Project 1 Captures/Ultra Wide Checkerboard/    33 ultra-wide calibration views
Project 1/Project 1 Captures/Straight Edge/              IMG_1125 (0.5x) and IMG_1126 (1x), metal ruler
Project 1/Homework 2 Image Captures/Checkerboard/        25 calibration views (--all only)
Project 1/Homework 3 Image Captures/                     Dark Frames/, White Frames/ (--all only)
Project 1/Homework 4 Image Captures/                     Blackbody/, LED/, Outdoor/ (--all only)
```

All captures are Bayer DNGs from Halide Mark III, and every series passes the course's `lens_id`
same-camera check.

## Tests

```bash
uv run pytest "Project 1/tests"
```

38 tests, no captures needed. They run the real analysis code on synthetic inputs with known
answers:

| File | What it checks |
|---|---|
| `test_mtf.py` | slanted-edge MTF of a Gaussian-blurred edge against the exact MTF; edge orientation and polarity |
| `test_noise.py` | gain and read noise recovered from simulated Poisson–Gaussian pairs; single-frame bias from PRNU; the 1/12 DN² rounding correction; dynamic range |
| `test_color.py` | white-balance gains, least-squares color matrix, the chart-orientation check (rotated and mirrored charts are rejected) |
| `test_camchar.py` | the uneven-light square finder, DNG opcode parsing, Apple's warp polynomial, the radial distortion profile, the straightness measure, the bootstrap helper, the notebook loader |

The homework notebooks' functions are tested in place: `camchar.notebook_functions` loads them by
name from the `.ipynb`, so there is one copy of each function.

## Repository layout

```
Project 1/
  Project 1 Report IEEE.ipynb    the report and the decisions-and-verification statement
  results.json                   main camera, course template ([value, uncertainty] pairs)
  results_ultrawide.json         ultra-wide, same template
  reproduce.py                   one-command reproduction
  project1_checks.py             the Project 1 measurements not in the homework notebooks
  camchar.py                     reusable, tested functions (square finder, DNG opcodes, straightness, ...)
  tests/                         pytest suite
  Measure Your Cameras Spatial Resolution.ipynb   Homework 2: calibration, undistortion, MTF
  Noise_and_Linear_Light.ipynb                    Homework 3: photon transfer, read noise, dark current
  Color and Accuracy.ipynb                        Homework 4: color correction and ΔE
  Setup and First Captures.ipynb                  Homework 1: raw capture and lens check
  figures/                       report figures
pyproject.toml, uv.lock          course environment (uv)
psu_imaging.py                   course helper module
```

## AI use

Claude (Anthropic) wrote most of the analysis code, `project1_checks.py`, `camchar.py`, the tests
and this README, and assembled the report, under my direction and review. The report's
decisions-and-verification statement describes what was checked and how.

## License

See [LICENSE](LICENSE).
