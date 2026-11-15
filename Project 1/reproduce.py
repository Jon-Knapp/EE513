"""Reproduce Project 1 from the raw captures with one command.

    uv run python "Project 1/reproduce.py"            checks, tests, results.json check, report
    uv run python "Project 1/reproduce.py" --all      also re-run the three homework notebooks first
    uv run python "Project 1/reproduce.py" --pdf      also print the report to PDF (needs Chrome)

Steps:
  1. confirm the raw captures are where the code expects them
  2. (--all) re-execute the homework notebooks in place: geometry and MTF (Homework 2),
     noise and dynamic range (Homework 3), color (Homework 4)
  3. run project1_checks.py: jackknife calibration, frame-corner MTF, ultra-wide calibration,
     Apple's opcode, metal-ruler test -> project1_checks.json and figures/
  4. run the pytest suite in tests/
  5. check every results.json entry against the template's [value, uncertainty] form, and the
     geometry, MTF and color entries against the values the code just computed
  6. execute the report notebook and export it to HTML (and PDF with --pdf)
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NOTEBOOKS = ["Measure Your Cameras Spatial Resolution.ipynb", "Noise_and_Linear_Light.ipynb",
             "Color and Accuracy.ipynb"]
REPORT = "Project 1 Report IEEE.ipynb"
NEEDED = {
    "Homework 2 Image Captures/Slanted Edge": "Homework 2 slanted-edge DNGs",
    "Project 1 Captures/Main Corner MTF": "frame-corner MTF DNGs (center, top-left, ... subfolders)",
    "Project 1 Captures/Ultra Wide Checkerboard": "ultra-wide checkerboard DNGs",
    "Project 1 Captures/Straight Edge": "metal-ruler DNGs IMG_1125 (0.5x) and IMG_1126 (1x)",
}
NOTEBOOK_DATA = ["Homework 2 Image Captures", "Homework 3 Image Captures", "Homework 4 Image Captures"]


def step(msg):
    print(f"\n=== {msg}", flush=True)


def run(cmd):
    print("$", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], cwd=HERE, check=True)


def check_captures(need_notebook_data):
    missing = [f"  {d}/  ({what})" for d, what in NEEDED.items()
               if not any((HERE / d).rglob("*.DNG"))]
    if need_notebook_data:
        missing += [f"  {d}/" for d in NOTEBOOK_DATA if not (HERE / d).is_dir()]
    if missing:
        sys.exit("Raw captures not found (they are not in git; see the README, 'Raw data'):\n"
                 + "\n".join(missing))
    print("all capture folders present")


def close(a, b, digits):
    return round(a, digits) == round(b, digits)


def check_results():
    R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    T = json.loads((HERE / "results.template.json").read_text(encoding="utf-8"))
    C = json.loads((HERE / "project1_checks.json").read_text(encoding="utf-8"))
    problems = [f"key missing from results.json: {k}" for k in T if k not in R]
    for k, v in R.items():
        if isinstance(T.get(k), list) and v is not None and not (
                isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)):
            problems.append(f"{k} is not a [value, uncertainty] pair: {v}")
    cal = C["calibration"]["five_coef"]
    expect = {  # results.json key: (value, uncertainty) as computed, decimals compared
        "focal_length_px": ((cal["f"]["value"], cal["f"]["jackknife_se"]), 1),
        "principal_point_x_px": ((cal["cx"]["value"], cal["cx"]["jackknife_se"]), 1),
        "principal_point_y_px": ((cal["cy"]["value"], cal["cy"]["jackknife_se"]), 1),
        "distortion_k1": ((cal["k1"]["value"], cal["k1"]["jackknife_se"]), 3),
        "distortion_k2": ((cal["k2"]["value"], cal["k2"]["jackknife_se"]), 3),
        "distortion_k3": ((cal["k3"]["value"], cal["k3"]["jackknife_se"]), 3),
        "distortion_p1": ((cal["p1"]["value"], cal["p1"]["jackknife_se"]), 5),
        "distortion_p2": ((cal["p2"]["value"], cal["p2"]["jackknife_se"]), 5),
        "reprojection_rms_px": ((cal["rms"]["value"], cal["rms"]["jackknife_se"]), 3),
    }
    for k, v in C["frame_corner_mtf"]["results_json"].items():
        expect[k] = (tuple(v), 3)
    color_file = HERE / "hw4_color_results.json"
    if color_file.exists():
        col = json.loads(color_file.read_text(encoding="utf-8"))
        expect["delta_e_mean_before_ccm"] = ((col["color_mean_dE_before"], R["delta_e_mean_before_ccm"][1]), 2)
        expect["delta_e_mean_after_ccm"] = ((col["color_mean_dE_after"], R["delta_e_mean_after_ccm"][1]), 2)
    for k, ((val, unc), d) in expect.items():
        got = R.get(k)
        if got is None or not (close(got[0], val, d) and close(got[1], unc, d)):
            problems.append(f"{k}: results.json has {got}, the code gives [{val:.{d}f}, {unc:.{d}f}]")
    checked = len(expect)
    if problems:
        print("\n".join("  " + p for p in problems))
        sys.exit(f"results.json does not match the computed values ({len(problems)} problem(s)).")
    print(f"results.json: every entry is well formed; {checked} entries match the computed values")
    print("  (noise and dynamic-range entries are read from the Homework 3 notebook's printed output)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="re-run the three homework notebooks first")
    ap.add_argument("--pdf", action="store_true", help="also print the report to PDF with headless Chrome")
    a = ap.parse_args()
    py = sys.executable

    step("1. raw captures")
    check_captures(a.all)
    if a.all:
        step("2. homework notebooks")
        for nb in NOTEBOOKS:
            run([py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                 "--ExecutePreprocessor.timeout=3600", nb])
    step("3. project1_checks.py")
    run([py, "project1_checks.py"])
    step("4. tests")
    run([py, "-m", "pytest", "-q", "tests"])
    step("5. results.json")
    check_results()
    step("6. report")
    run([py, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", REPORT])
    run([py, "-m", "jupyter", "nbconvert", "--to", "html", "--no-input", REPORT])
    if a.pdf:
        chrome = next((shutil.which(c) for c in ("chrome", "google-chrome", "chromium", "msedge")
                       if shutil.which(c)), None)
        if chrome is None:
            print("no Chrome or Edge on PATH: open the HTML and print it to PDF instead")
        else:
            html = (HERE / REPORT).with_suffix(".html")
            run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                 f"--print-to-pdf={html.with_suffix('.pdf')}", html.as_uri()])
    print("\nDone: project1_checks.json, figures/, and the report HTML are up to date.")


if __name__ == "__main__":
    main()
