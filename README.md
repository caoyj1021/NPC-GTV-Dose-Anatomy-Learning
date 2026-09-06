# GTV-informed three-dimensional dose–anatomy learning for severe acute oral mucositis in nasopharyngeal carcinoma

Version 1.2.0 of the public analysis-code release accompanying the manuscript.

Public repository: https://github.com/caoyj1021/NPC-GTV-Dose-Anatomy-Learning  
Archived release: Zenodo DOI: https://doi.org/10.5281/zenodo.22459984

## What this repository contains

- preprocessing and spatial input construction for planning CT, RT dose, oral-cavity masks, and GTV masks;
- repeated five-fold split generation and 3D CNN training for M1, M2-Abl, M2, M3-Abl, M3, and M4;
- locked independent external inference and statistical evaluation;
- manuscript Figure 1–4 generation from the required local source arrays/workbooks.

Patient images, contours, clinical data, model checkpoints, prediction files, and other governed research data are not distributed.

## Cohort-version provenance

Three numbers have different meanings and must not be interchanged:

1. **497-image preprocessing candidate pool.** Imaging was preprocessed before the final distant-metastasis exclusions. Legacy NPZ files may therefore contain an obsolete `cohort` field from the early A+B development/C external plan. Downstream code deliberately ignores that field.
2. **309-patient development cohort.** Final model development used Centers B+C only. Center A was not loaded for model fitting, early stopping, model selection, threshold selection, or calibration.
3. **489-patient final analysis cohort.** The final locked master excludes 51 of 540 assessed patients, leaving B+C=309 for development and A=180 for independent external validation.

Split generation, training, external validation, and figure-generation code now all filter the frozen master on blank `exclude_reason` and assert 489/309/180. The 497-file NPZ pool is treated only as an upstream image pool; the eight additional Center-A records are verified as excluded in the final master and cannot enter split assignments, training, or evaluation. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## PR-AUC definition

All reported PR-AUC values in this release use **trapezoidal integration of the precision–recall curve with recall on the x-axis** (`precision_recall_curve` followed by `auc(recall, precision)`). This is intentionally different from scikit-learn's step-weighted average precision. The manuscript, tables, and figures use the trapezoidal definition.

## Setup

Python 3.10 is recommended. Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate npc-gtv-dose-anatomy-learning
```

For GPU training, install the PyTorch build appropriate for the local CUDA driver if the resolver does not select it automatically. Record the final environment with `python -m pip freeze` for an archival rerun.

Set the paths and private case-level audit values shown in `config.example.env` in the environment before opening Jupyter. The notebooks reject missing or relative root paths instead of silently treating placeholder text as a directory.

## Recommended execution order

1. Optional historical image-pool reconstruction: `preprocessing/generate_preprocessed_dataset.ipynb`. This step requires the historical 497-candidate preprocessing master and may be skipped when audited candidate-pool NPZ files already exist.
2. `training/generate_cross_validation_splits.ipynb`
3. `training/train_spatial_dose_anatomy_models.ipynb`
4. `validation/external_validation_inference.ipynb`
5. `qc/patch_coverage_qc.ipynb`
6. `evaluation/calibration_and_dca_analysis.py`
7. `interpretability/generate_layer4_gradcam.ipynb`
8. `evaluation/figure1_generate.ipynb` through `figure4_gradcam_generate.ipynb`

Figure 2 recomputes ROC AUC and trapezoidal PR-AUC directly from patient-level predictions; it does not trust a potentially stale metrics CSV. The calibration/DCA script regenerates the complete Figure 3 analysis workbooks from frozen patient-level predictions. The Grad-CAM notebook generates Layer4 maps from locked M3 checkpoints; Figure 4 then performs the fixed anatomy/CAM slice-ranking and final assembly.

## Data contract

See [DATA_SCHEMA.md](DATA_SCHEMA.md). All identifiers must be pseudonymous. Do not commit governed data or derived patient-level files; `.gitignore` blocks common image, array, checkpoint, spreadsheet, and prediction formats by default.

## Reproducibility boundary

The repository exposes analysis logic but cannot reproduce numerical manuscript results without the governed frozen master, preprocessed arrays, checkpoints, and prediction artifacts. Exact software versions and CUDA/cuDNN information from the original run were not recoverable from the supplied bundle; dependency ranges are therefore compatibility ranges, not a claim of bitwise reproduction.

## License and citation

Code is released under the MIT License. See `LICENSE` and `CITATION.cff`. The frozen v1.2.0 release is archived in Zenodo at DOI [10.5281/zenodo.22459984](https://doi.org/10.5281/zenodo.22459984). README-only metadata updates on the main branch do not alter the archived release.
