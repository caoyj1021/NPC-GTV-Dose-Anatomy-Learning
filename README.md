# NPC GTV dose–anatomy learning

Release source **1.3.0**. This version includes the analyses added during manuscript revision.

The executable workflow uses one final cohort throughout: **540 assessed, 51 excluded, 489 included; B + C development 309; A independent external validation 180**. A = Jiaxing, B = Yantai, C = Ningbo. Inclusion is determined by the final master, not the number of files in an image folder or embedded NPZ labels. See [Implementation details](#implementation-details) for the implemented methods and scope, and [Input data schema](#input-data-schema) for required fields.

## Configure once

1. Keep the repository outside the governed-data/results directories.
2. Use the existing compatible PyTorch environment, or follow [Environment](#environment). Install missing dependencies in the selected Jupyter kernel; do not replace a working CUDA installation solely to organize code.
3. Copy `config.example.json` to `config.local.json`. Set the absolute `master`, `master_sheet`, `npz_dir` and `result_root` paths. Alternatively, set `NPC_CONFIG` to the absolute path of a private configuration file.
4. Start Jupyter from this repository and open `00_start_here.ipynb`. The notebooks share functions in `npc_pipeline`; do not copy individual notebooks out of the repository.

Every preparation, split, training or analysis run creates a new timestamped result directory. The local configuration records the successful output paths for downstream steps. Explicit CNN resumption continues a named run only after matching its inputs, settings and saved file hashes. Use one workflow at a time per configuration file; use separate configurations for concurrent independent work.

## Replay the completed study without retraining

This is the recommended route for checking the completed study after code organization.

1. Configure the final integrated master, the existing NPZ directory and `existing_splits_root`. The existing split root may contain `repeat_XX/fold_XX/locked_split_inputs`; only the development split CSVs and seeds are imported.
2. Run `qc/npz_integrity_audit.ipynb` if the arrays have not recently been checked.
3. Run `training/generate_cross_validation_splits.ipynb`. It verifies every imported inner/outer patient partition against the final fixed algorithm and writes a new final-cohort split lock.
4. Set `frozen_sources.development_predictions`, `external_predictions`, `revision_runs.MasksOnly`, `revision_runs.DoseGTV`, and `clinical_run` to the completed analyses. Set `frozen_sources.cnn_root` if Grad-CAM is needed.
5. Run `evaluation/export_frozen_predictions.ipynb`. It checks the actual IDs, outcomes, centers, ensemble counts, four OOF assignments, OOF means, and locked folds, then produces a common prediction layout.
6. Run calibration/DCA, paired comparisons and the additional center/volume/multiplicity analyses. Figure 2 and Figure 3 consume these outputs.
7. For Grad-CAM, provide the frozen development threshold and approved illustrative cases. Run the extraction notebook and then Figure 4. Development cases use their four held-out checkpoints; external cases use all 20. Selection of the displayed sagittal slice uses the original anatomy/CAM composite rule.

The master path may contain additional clinical fields without changing the 309 development identities. Existing NPZ directories may contain unselected candidate files. Neither fact changes the final analysis cohort. Extra NPZ files are recorded and ignored; embedded NPZ outcome/cohort fields are not used for learning.

## Run the final-cohort pipeline from source images

1. Use a 540-row final master with the required schema. Record per-case dose conversion factors in `dose_scale_csv`; no unit inference from dose magnitude is performed.
2. Run `preprocessing/generate_preprocessed_dataset.ipynb`, followed by `qc/patch_coverage_qc.ipynb` as needed. Unresolved crop warnings are reported; they do not silently exclude patients. Reuse clinical approval records only when the measured retention fraction matches.
3. For clinical predictors, `preprocessing/extract_gtv_dvh.ipynb` extracts native-mask GTV volume and oral DVH and exports a separate workbook in master order. Review and merge by `patient_id` into the analysis master; the script does not overwrite the master. Do not substitute cropped/resampled GTV volume for native-mask volume.
4. Generate the fixed 4 × 5 development splits. The outer seeds are 20260701–20260704. Inner seed = 50260700 + repeat × 10 + fold; training seed = 30260700 + repeat × 10 + fold. Order before splitting is part of the method.
5. `training/train_spatial_dose_anatomy_models.ipynb` starts the six original CNN configurations. The two notebooks under `revision/` train masks-only and dose+GTV separately using the same core and splits. These are full training jobs; running them is not necessary to preserve the completed study.
6. Complete and freeze development training before `validation/external_validation_inference.ipynb`. No tuning, threshold optimization or recalibration uses external performance.
7. Export frozen predictions. Run C1/C2 if required, then export again so all ten models are available. Run the statistical analyses and figure notebooks. Re-exporting predictions requires regeneration of calibration/DCA and Grad-CAM inputs before their figures; artifact locks reject mismatched inputs.

Newly retrained models need not reproduce published probabilities bit for bit. Fixed seeds and deterministic settings do not remove all GPU/operator and library-version variability. A newly selected development threshold must be frozen before external evaluation; the example threshold belongs to the original frozen M3 only.

## Models and methods

| Model | Image channels / predictors |
|---|---|
| M1 | dose |
| M2-Abl | dose, oral |
| DoseGTV | dose, GTV |
| MasksOnly | oral, GTV |
| M2 | dose, oral, dose×oral |
| M3-Abl | dose, oral, GTV |
| M3 | dose, oral, GTV, dose×oral, dose×GTV |
| M4 | CT, dose, oral, GTV, dose×oral, dose×GTV |
| C1 | age, BMI, sex, smoking, drinking, T/N stage, actual induction/concurrent chemotherapy, oral Dmean and V30 |
| C2 | C1 + ln(1 + primary-GTV volume in cm³) |

All image models use the same oral-centered patch. The dose+GTV model therefore still inherits oral localization. No scalar clinical branch is added to the CNNs.

The shared CNN core retains the lightweight ResNet-10, base width 16, dropout 0.20, batch size 4, AdamW and fixed 120-epoch absolute learning-rate schedule. Stage A selects an epoch from smoothed inner BCE loss; Stage B starts afresh on the full outer training set. Reporting PR-AUC uses trapezoidal integration with recall on the x-axis; it is not average precision. Clinical preprocessing and regularization selection are fitted within the development splits. Per-run CNN/clinical summaries are descriptive diagnostics. Supplementary Tables S1–S2 use the harmonized ordinary patient bootstrap in `paired.py`; Table S5 uses the separate stratified 20,000-resample sensitivity analysis in `additional_analysis.py`. These must not be substituted for the original Table 2 stratified intervals.

`evaluation/figure1_generate.ipynb` exports an archived raster illustration. It is not an editable-vector or patient-analysis generator. Figure 2–4 are generated from governed analysis inputs. The package does not distribute patient data, trained checkpoints, image arrays or per-patient predictions.

## Environment

Engineering and frozen-data checks were run on 27 October 2026 with Python 3.13.9, PyTorch 2.7.1+cu118, CUDA build 11.8 and an NVIDIA GeForce RTX 4060 Ti. The cuDNN integer version queried during those checks was 90100 (9.1.0). This query does not retrospectively establish the cuDNN version used during the original training.

`requirements-tested.txt` pins the tested scientific stack, including pytest. `requirements.txt` provides broader dependency bounds. The pinned PyTorch build targets CUDA 11.8; another platform or accelerator may require a compatible PyTorch installation. Create a separate environment if needed; do not change an existing working GPU environment merely to inspect the source.

For a separate Conda environment, run these commands from the repository root:

```text
conda create -n npc-final489 python=3.13.9 pip
conda activate npc-final489
python -m pip install -r requirements-tested.txt jupyterlab
python -m pytest tests -q
jupyter lab
```

An existing compatible Jupyter kernel can run the notebooks directly from the repository. If the Python executable differs from the kernel, install missing dependencies through that kernel's `%pip`, not an unrelated terminal's pip.

The full training configuration uses deterministic seeds and `warn_only=True` deterministic-algorithm checks. Some CUDA operations may warn about nondeterminism. Numerical agreement is not guaranteed bit for bit on other software/hardware. CPU inference or Grad-CAM may use different precision from the recorded CUDA mixed-precision inference.

Patient-level files, checkpoints and the frozen clinical master are required for study reproduction and are not publicly included. Synthetic tests require no study data. Figure 1 is a packaged static illustration; all other empirical figures need the governed predictions or images described below.

## Input data schema

The master contains 540 assessed rows and a unique integer `patient_id`. Inclusion is `exclude_reason` empty. The 51 nonempty reasons are `M1 metastatic disease` (28), `missing CT RTPLAN` (20), and `no_mask_gtv` (3). Required core fields are `patient_id`, `model_center`, `severe_mucositis`, and `exclude_reason`. If `cohort` is supplied, its included-patient values must agree with A = External and B/C = Development; inconsistent values cause an error.

The included center × outcome counts are A 88/92, B 42/39, C 116/112 (outcomes 0/1). Center A is Jiaxing, B Yantai, C Ningbo. These are study-reproduction checks, not a general-purpose model-development template for new populations.

For source imaging, provide `ct_path`, `dose_path`, `oral_cavity_mask_path`, and `gtv_mask_path`. Each may be an explicit file or a directory containing exactly one `<patient_id>.nii.gz` or `<patient_id>.nii`. Ambiguous and missing paths stop the case. GTV refers to the actual primary-tumor mask; this workflow does not reconstruct GTV from other targets. Dose refers to the first radiotherapy plan used by the study.

`dose_scale_csv` has unique integer `patient_id` and positive `dose_scale_to_gy` (e.g. 1 for source Gy, 0.01 for cGy after independent unit verification). Do not fill values based only on dose magnitude. This file is private. `coverage_approvals_csv`, when needed, has `patient_id`, `approved` (1 for documented approval), and the reviewed `gtv_retained_fraction`; a changed fraction is not automatically reapproved.

NPZ files are named by `patient_id` and contain `ct`, `dose`, `oral`, `gtv`, and scalar `patient_id`. Shape is ZYX (64,112,80). CT is clipped from −1000 to 2000 HU and scaled to [−1,1]; dose is Gy/70 clipped to [0,1.2]; masks are binary. Grad-CAM also requires `patch_direction`, `dose_normalization_gy`, `dose_scale_to_gy`, and `dose_normalized_max` metadata. Embedded cohort/outcome metadata never overrides the final master.

The clinical master also contains `age`, `BMI`, `sex(female=0,male=1)`, `吸烟史`, `饮酒史`, `T_stage`, `N_stage`, `induction_chemotherapy`, `concurrent_chemotherapy`, `oral_Dmean_Gy`, `oral_V30_percent`, `gtv_volume_cm3`, and `extraction_status` = PASS for all included patients. Binary fields are actual 0/1 values; missing treatment information must not be replaced with zero. T/N are the confirmed eighth-edition diagnostic stages. Chemotherapy regimen text is not used as a model predictor. Actual concurrent treatment is a treatment-background variable and may not be available at an entirely pretreatment prediction time.

The center/year/volume analysis additionally requires `RT_start_year`. V30 is a percentage from 0 to 100, not a 0–1 fraction. Native primary-GTV volume is in cm³; C2 applies ln(1 + volume) internally.

`gradcam_cases_csv` contains `cohort` and `patient_id` (or `patient_key`), with optional `category`. Figure 4 uses the `figure4_cases` identifiers in the private configuration. Do not upload those case-selection files or generated patient-level artifacts with the public source repository.

## Implementation details

This release source implements the final analysis cohort and the revision-added analyses. The table links the executable workflow to the manuscript methods and defines which upstream processes are supplied as inputs.

| Component | Implementation and scope |
|---|---|
| Eligibility | 540 assessed; 51 excluded; 489 included. Exclusion counts are 28 distant metastases, 20 missing CT/RTPLAN and 3 unavailable GTV masks. |
| Centers | A = Jiaxing, 180 external patients (88/92 outcomes 0/1); B = Yantai, 81 development patients (42/39); C = Ningbo, 228 development patients (116/112). |
| Spatial inputs | Supplied CT-aligned NIfTI files; first-plan dose, oral mask and primary-tumor GTV mask. Clinical contour production, nnU-Net training, DICOM conversion, grading and clinical review are upstream processes, not performed by these notebooks. |
| Patch | Oral-bounding-box-centered 80 × 112 × 64 XYZ at 2 × 2 × 3 mm; ZYX array 64 × 112 × 80. Linear CT/dose and nearest-neighbor mask interpolation; CT −1000–2000 HU scaled to [−1,1]; dose Gy/70 clipped to [0,1.2]. |
| CNNs | Eight channel configurations share the same lightweight ResNet-10, Group Normalization, base width 16 and dropout 0.20. Every configuration retains oral-centered cropping. Clinical scalars are not CNN inputs. |
| Training | Batch 4, AdamW, BCE with logits, no random augmentation, weight decay 10⁻⁴, gradient clipping 5. Five-epoch warmup to 10⁻³ and fixed 120-epoch cosine schedule to 5 × 10⁻⁵. Stage A uses the three-epoch mean inner loss; eligibility starts at epoch 10, early stopping at epoch ≥40, patience 25 and minimum improvement 10⁻⁴. Stage B reinitializes and refits full outer training for the selected duration. |
| Validation | Four center/outcome-stratified five-fold repeats. Inner validation is 20% of outer training. Development averages four OOF probabilities per patient; external validation averages 20 frozen models. Epoch choice uses inner loss; external performance is not an input to fitting or selection. |
| Clinical C1/C2 | C1 includes age, BMI, sex, smoking/drinking, diagnostic AJCC 8th-edition T/N, actual induction/concurrent receipt, oral Dmean and V30. C2 adds ln(1 + full-source primary-GTV cm³). Training-only scaling/one-hot encoding and inner log-loss selection over 13 C values from 10⁻³ to 10³. Treatment receipt is a retrospective background variable, not necessarily known pretreatment. |
| Main and supplementary statistics | AUROC, trapezoidal PR-AUC and Brier use frozen patient-level probabilities. `paired.py` generates the harmonized 2,000 ordinary patient-bootstrap comparisons for Tables S1–S2. `additional_analysis.py` implements the separate 20,000 paired stratified bootstrap and 24-test Holm sensitivity family, plus center/year and volume analyses. Original Table 2 stratified confidence intervals and Table 1 descriptive formatting are not generated by the per-run diagnostic summary routines. |
| Calibration/DCA | M3: 10 qcut groups, Wilson intervals, binomial GLM slope/intercept, CITL with logit offset; clip to [10⁻⁶,1−10⁻⁶]. DCA evaluates 0.01–0.99. No external recalibration. Figure 3 displays a restricted threshold window, without changing the analysis grid. |
| Grad-CAM | Positive binary logit, Layer4 activations/gradients, ReLU and trilinear upsampling. Development uses four held-out models and external uses 20 models. Maps are illustrative and case-normalized. They do not establish causality or voxel-level localization. |
| Figure 4 slice | Both oral and GTV visible; descending CAM gates 0.60/0.50/0.40/0.30/0.20 until ≥3 candidates, otherwise dual-structure candidates without a CAM gate. Score = 0.55 × minimum normalized ROI area + 0.25 × geometric mean ROI area + 0.20 × normalized integrated CAM. Tie order: anatomy balance, CAM fraction, then slice index. This is display selection only. |
| Figures and scope | Figure 1 exports the supplied static illustration. Figures 2–4 render governed predictions or arrays. The repository does not automate the entire Word manuscript, treatment-label adjudication, clinical contour approval or every source-table formatting step. |

The exploratory comparisons do not establish superiority to all conventional models, a separate dose×GTV effect, or a definitive M3 improvement over masks alone.

Each new run has a separate output directory. Calibration, DCA and Grad-CAM output locks connect the generated files to the current master, split lock and prediction audit; altered or mismatched inputs require regeneration.

## Verification

Run `python -m pytest tests -q` from the repository root.

The following checks were completed on 27 October 2026 in the environment described above.

- 28 synthetic regression tests passed. Three SimpleITK/SWIG deprecation warnings were reported.
- All 20 Jupyter notebooks compile and contain no execution outputs. All Python source files compile.
- The final master passed 540/51/489 and center/outcome checks. All 489 selected NPZ inputs passed identity, shape, finite-value, mask and intensity-range checks.
- The 20 fixed folds and all 80 inner/outer partitions match the completed study's development splits.
- All ten models aligned to 309 development and 180 external patients. C1/C2 were additionally checked against every OOF fold, each external ensemble member and the corresponding four-/20-model averages.
- Calibration and DCA were regenerated; all six workbook sheets match the checked frozen analyses within 10⁻¹² tolerance.
- The harmonized bootstrap regenerated 80 performance rows and 104 paired-difference rows with identical numerical results.
- Figure 2, Figure 3, Layer4 Grad-CAM extraction and Figure 4 rendering completed. Figure 4 used the manuscript's configured cases, four development held-out checkpoints and 20 external checkpoints. The two slice/dose audit rows match the checked analysis.
- New tests reject mismatched downstream inputs and modified output files, fractional/multivalued NPZ identifiers and inconsistent clinical ensemble means. The optional checkpoint-root fallback accepts the documented null configuration.

A local comparison audited 64 definitions against the source implementations: 60 retain identical abstract syntax trees (ASTs). Detailed engineering audit records are retained locally. Adaptations affect input identity validation, output naming and native-DVH unit/row bookkeeping. The CNN architecture, optimizer/learning-rate schedule, two-stage fitting and epoch selection, clinical fitting, metric definitions, bootstrap formulas and Figure 4 slice score were not retuned.

These checks use frozen study inputs and predictions, plus synthetic tests. They do not repeat every full CNN or C1/C2 fit, establish bitwise GPU reproducibility, or verify installation on a new computer. The package contains no patient-level imaging, master workbook, predictions, model checkpoints or private configuration. Release source 1.3.0 was prepared on 28 October 2026 from the checked 1.3.0-rc3 package. Only version metadata and release wording changed at that step. For the compact upload package prepared on 2 November 2026, documentation was consolidated into this README; the Chinese user guide, detailed local audit reports and redundant environment/test-install wrappers are not included. All analysis implementations, notebook source and synthetic tests remain byte-for-byte identical to the checked release source. Archival status must be verified separately.

## Citation

For citation, use the version-specific Zenodo DOI assigned to this release after archival. Software metadata are provided in `CITATION.cff`.
