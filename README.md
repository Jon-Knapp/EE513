# EE 513 Student Environment

**Purpose:** the two files students download from Canvas to build the course Python environment.

**Built and verified:** 2026-09-29

**Post these two files, and nothing else:**

- `pyproject.toml`
- `uv.lock`

That is the whole payload. Tested from an empty folder containing only those two: `uv sync` picked Python 3.12.12 on its own, created `.venv`, and installed 116 packages. Students do not need a `.python-version` file, an installer, or admin rights.

---

## What this is and is not

This is the **EE 513 only** environment. It is deliberately smaller than the environment on your own machine, which covers all three courses at 213 packages and 2.2 GB. Open3D, GTSAM, pyceres, and evo are EE 515 tools and are not here.

Your own environment is the one at the `Courses` root. Do not point students at it, and do not merge the two; see `Courses/Python-Environment.md`.

## It installs in two stages

The lock holds 174 packages, but no student installs all of them in week 1.

| Stage | Command | Packages | Installed |
|---|---|---|---|
| Week 1, core | `uv sync` | 116 | 643 MB |
| Week 6, adds PyTorch | `uv sync --group learning` | 152 | 1329 MB |

PyTorch, torchvision, and gradio are in a `learning` dependency group rather
than the core list. They are 686 MB and 36 packages that nothing before lecture
11 imports, and on Linux the torch wheel is the CUDA build, which is far larger
again. Week 1 is when students are most likely to get stuck and least able to
diagnose it, so it carries the least possible load. Measured 2026-09-29.

**The one sharp edge.** From week 6 onward, `--group learning` is part of the
command. A bare `uv sync` makes the environment match the core set exactly,
which means it **removes torch**. That is uv working as designed, not a bug, but
a student will read it as one. It is called out in the Canvas page and in the
troubleshooting list; say it out loud in lecture 11 as well.

To undo the split, move the three packages back into `[project.dependencies]`,
delete the `[dependency-groups]` block and the `default-groups` line, and run
`uv lock`. Nothing else depends on it.

## Keeping it current

If you add a package mid-term, do it here and re-post both files:

```bash
cd "EE 513 IP/Student Environment"
uv add some-package                      # into the week 1 core
uv add --group learning some-package     # into the week 6 group
```

`uv add` edits `pyproject.toml` and rewrites `uv.lock` together, so the pair stays consistent. Announce it, because students have to re-run `uv sync` to pick it up.

To let students run something once without changing the environment at all, have them use:

```bash
uv run --with some-package python script.py
```

That is the better answer for a one-off in a single lecture.

## What is in it, and why

| Package | Why |
|---|---|
| numpy, scipy, matplotlib, pandas | the whole term |
| opencv-contrib-python | **contrib, not plain opencv-python**, see below |
| scikit-image, imageio, tifffile, pillow | image IO and classical algorithms |
| rawpy | Project 1 captures and Project 2 raw decoding |
| colour-science | lecture 6, color matching functions and Delta E |
| torch, torchvision | **`learning` group**, week 6: lecture 11 onward and the Project 2 learned denoiser |
| gradio | **`learning` group**, week 6: demo front ends for project deliverables |
| jupyterlab, ipykernel, ipywidgets | the demo notebooks |
| pytest | projects require tests |

**The contrib package matters.** Plain `opencv-python` 5.0 does not ship `HOGDescriptor`, `CascadeClassifier`, `AKAZE`, `KAZE`, or `BRISK` at all. `opencv-contrib-python` 5.0 restores `cv2.HOGDescriptor` and `cv2.CascadeClassifier` to the flat namespace and puts the feature detectors under `cv2.xfeatures2d`. Verified on both packages 2026-09-29. A student who installs plain `opencv-python` by hand will hit missing attributes, so the lock is what keeps this right.

## Cross-platform

`uv.lock` is universal, not macOS-specific. It carries wheel references for macOS arm64 and x86_64, Linux x86_64 and aarch64, and Windows x86_64, so the same two files work for every student regardless of machine. Linux students get the CUDA build of torch from PyPI, which is large but correct.

## The Canvas page

`Canvas Page.html` is the student-facing instructions, ready to paste into a Canvas page using the HTML editor (the `</>` button in the rich content editor). Attach `pyproject.toml` and `uv.lock` to the same module.
