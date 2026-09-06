# Reproducibility and audit notes

## Frozen analysis definitions

| Stage | Population | Role |
|---|---:|---|
| Assessed master | 540 | Starting registry |
| Excluded | 51 | 28 M1 metastatic disease, 20 missing CT/RTPLAN, 3 missing GTV mask |
| Final analysis | 489 | 246 outcome-negative, 243 outcome-positive |
| Development | 309 | Centers B+C; repeated 4 × 5-fold evaluation |
| Independent external validation | 180 | Center A; locked until model development was complete |

## Why some paths still contain “497”

The directory names containing `497` identify the historical preprocessing candidate pool and the downstream folders created from it. They are retained because existing split records, checkpoint paths, and hashes refer to those names. They do **not** define the final analytic sample size.

The preprocessing candidate pool was originally labelled A+B development/C external. The final modelling plan redefined B+C as development and A as external. Split generation and training derive the analytic role from `model_center`; they do not use the legacy NPZ `cohort`, NPZ label, or preprocessing index label.

The executable split/training contract is now 540 assessed, 51 excluded, 489 included, B+C=309 development, and A=180 external. The 497-file preprocessing pool contains eight additional Center-A records that are marked excluded for distant metastasis in the final master. Code verifies this relationship and restricts every analytic assignment to the final 489 IDs. Because the B+C development IDs and all 20 outer/inner folds are unchanged, no model retraining is required to preserve the published checkpoints.

## Metric lock

- ROC AUC: standard area under the ROC curve.
- PR-AUC: trapezoidal area under precision versus recall, recall on the x-axis.
- `average_precision_score` is not used for manuscript PR-AUC.
- Changing the reporting implementation does not change model checkpoints because checkpoint selection used validation loss, not PR-AUC.

## Patch coverage

The final 489-case audit processed all cases without error. Oral-cavity retention was 1.000 for every case. GTV retention was at least 0.99 for 487 cases; two cases had retained fractions 0.986317 and 0.986785, both above 0.95 and both passed subsequent manual review.

## Expected non-public inputs

- historical preprocessing-source master, only when reconstructing the 497-image candidate pool;
- frozen final-489 analytic master for all downstream cohort, label, and split assignments;
- 497 candidate-pool NPZ files (only final-master IDs are used downstream);
- locked split artifacts and trained checkpoints for exact validation;
- patient-level OOF and external prediction files;
- patient-level prediction files used to regenerate calibration/DCA workbooks;
- trained checkpoints and a private case-selection CSV used to regenerate Grad-CAM arrays.

These files are not included because they contain governed data, pseudonymous patient-level records, or large derived artifacts.
