# Far-OOD detection and calibration on BCN20000: a pre-registered evaluation

Code, per-seed results, and analysis outputs for a pre-registered evaluation of whether far-out-of-distribution (far-OOD) signal in dermoscopic classifiers survives brightness and contrast interventions. The main outcomes are mostly inconclusive under the registered decision rules. A manuscript describing the study is in preparation.

## Contents

| Path | Contents |
|---|---|
| `docs/dose_response_design.md` | Pre-registered protocol (sections 14, 16, 17 cover the analyses reported here). Sections 14 and 17 contain clarifications written after some results; the timing of each section is stated in the manuscript. |
| `code/ood_calibration.py` | Training, evaluation, calibration, and OOD scoring (EfficientNet-B0 and ConvNeXt-T). |
| `code/analyze_e.py` | Decision tables for E1 (resolution), E2 (placebo), E3 (PAD-UFES-20), and E5 (20 seeds). |
| `code/analyze_jit.py` | Gates (C0, G1, G2, G3), R ratio with both readings, decision table, and K1 (within-bin check). |
| `code/make_farsub.py` | Builds the far-OOD subsets used for training and evaluation (requires MedMNIST+ files, see Data). |
| `code/make_notebook_kaggle.js` | Generates the Kaggle notebooks in `notebook/`. |
| `kaggle_output/ma21_bcn20000/<run>/outputs/` | Per-seed results (`results_seed*.json`), predictions, far-OOD scores, gray-control results, and per-seed summaries. Checkpoints are not included. |
| `analysis/` | Decision tables and intermediate analysis outputs. |

Run folders and code versions (each run's `run_info.json` records its script hash, except `v3r2`, whose version is given in the project log):

- `v3b`, `v3j`: earlier code version (SHA 009989…), used as B and J in the E2 and E5 seeds 0–9 analyses.
- `v3b2`, `v3j2`, `v3p`, `v3pb`, `v3pj`, `e4b`, `e4j`: current code version (SHA f13889…). `v3r2` (E1) is also the current version per the project log.
- `v3h`, `v5h` (SHA bf3bf1…) and `v5g` (SHA e94936…): earlier grayscale and learning-rate runs. Their decisions did not pass the registered gates, and they are not part of the confirmatory analyses above.

## Data

Image data are not included. Obtain them from the original sources:

- **BCN20000** (ID training set and CSV): Hernández-Pérez C, et al. *Scientific Data* 2024. DOI [10.1038/s41597-024-03387-w](https://doi.org/10.1038/s41597-024-03387-w). Figshare record: DOI 10.6084/m9.figshare.24140028.
- **PAD-UFES-20** (second ID set, E3): Pacheco AGC, et al. *Data in Brief* 2020. DOI [10.1016/j.dib.2020.106221](https://doi.org/10.1016/j.dib.2020.106221). Mendeley Data DOI [10.17632/zr7vgbcyr2.1](https://doi.org/10.17632/zr7vgbcyr2.1).
- **MedMNIST+** (far-OOD sets, 224 px): Zenodo record [10519652](https://doi.org/10.5281/zenodo.10519652). Benchmark: Doerrich S, et al. *Scientific Reports* 2025. DOI [10.1038/s41598-025-92156-9](https://doi.org/10.1038/s41598-025-92156-9).

Far-OOD subsets: RetinaMNIST and PneumoniaMNIST and BreastMNIST are used in full (test split). PathMNIST and BloodMNIST are sampled to 800 images with `numpy.random.default_rng(0)` in `code/make_farsub.py`.

## Reproducing the analyses

The decision tables can be reproduced from the included outputs:

```bash
python code/analyze_e.py res   --b kaggle_output/ma21_bcn20000/v3b/outputs --r kaggle_output/ma21_bcn20000/v3r2/outputs --out analysis/e1_res
python code/analyze_e.py rot   --b kaggle_output/ma21_bcn20000/v3b/outputs --j kaggle_output/ma21_bcn20000/v3j/outputs --p kaggle_output/ma21_bcn20000/v3p/outputs --out analysis/e2_rot_kaggle
python code/analyze_e.py pad   --j kaggle_output/ma21_bcn20000/v3pj/outputs --b kaggle_output/ma21_bcn20000/v3pb/outputs --out analysis/e3_pad_kaggle
python code/analyze_e.py seeds --j kaggle_output/ma21_bcn20000/v3j/outputs --b kaggle_output/ma21_bcn20000/v3b/outputs --j2 kaggle_output/ma21_bcn20000/v3j2/outputs --b2 kaggle_output/ma21_bcn20000/v3b2/outputs --out analysis/e5_seeds_kaggle
python code/analyze_jit.py --j kaggle_output/ma21_bcn20000/e4j/outputs --b kaggle_output/ma21_bcn20000/e4b/outputs --out analysis/e4_convnext
```

Notes:

- `analyze_jit.py` prints the gates and decision table, then runs K1. K1 needs the far-OOD subsets (from Data above) and the cached training images (`X_train_256.u8`, built from the BCN20000 images). Without them, K1 fails after the decision table is printed.
- Three code files (`code/analyze_jit.py`, `code/diag_brightness.py`, `code/make_farsub.py`) contain the author's local paths as constants (for example `D:/NCKH_3/...` and `C:/data/bcn20000`). Edit them to your own paths before running. They are left unchanged here so that the script hashes match the runs.
- Training was run on Kaggle (2 × NVIDIA T4) with PyTorch 2.11 (CUDA 12.8), one seed per GPU.

## Licence

- Code: MIT (see `LICENSE`).
- Results and analysis outputs: CC BY 4.0. The source datasets have their own licences; check them before reuse.

## Archive and citation

The v1.0.0 archive is deposited on Zenodo: https://doi.org/10.5281/zenodo.23723362 (concept DOI https://doi.org/10.5281/zenodo.23723361 covers all versions).

Trương Đình Phúc (2026). *Pre-registered far-OOD evaluation on BCN20000: code, per-seed results, and analysis outputs* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.23723362
