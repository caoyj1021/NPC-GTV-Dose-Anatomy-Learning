# Data schema

## Frozen final analytic master workbook

This workbook is authoritative for the final 489-patient cohort, labels, model-center roles, split generation, training, and evaluation. The optional historical preprocessing notebook instead uses the separately frozen 497-candidate preprocessing-source master described in `REPRODUCIBILITY.md`; its legacy cohort and outcome fields are not authoritative downstream.

Required fields used by the public notebooks:

| Field | Type | Meaning |
|---|---|---|
| `patient_id` | integer/string | Site-governed pseudonymous identifier; unique |
| `model_center` | categorical | `A`, `B`, or `C` |
| `severe_mucositis` | binary integer | 1 for Grade ≥3 acute oral mucositis, otherwise 0 |
| `exclude_reason` | text/blank | Blank means included in the final analysis |
| `ct_path` | path | Planning CT series or converted volume |
| `oral_cavity_mask_path` | path | Oral-cavity mask |
| `gtv_mask_path` | path | GTV mask |

The preprocessing notebook contains additional project-specific input-column checks. Adapt column aliases only after documenting the mapping.

## Preprocessed NPZ

Required arrays/metadata include `ct`, `dose`, `oral`, `gtv`, `patient_id`, and `model_center`. Expected array order is Z–Y–X and the project patch shape is 64 × 112 × 80, corresponding to 80 × 112 × 64 in X–Y–Z.

Legacy `cohort`, `label`, or `severe_mucositis` values stored in an NPZ are not authoritative. Final cohort membership and labels must come from the frozen master.

## Prediction files

Development patient-averaged OOF files require `patient_id`, `y_true`, and `oof_probability_mean`. External long-format predictions require `display_name`, `patient_id`, `y_true`, and `probability`.

Never replace pseudonymous IDs with medical-record numbers, names, accession numbers, or DICOM UIDs in files intended for public release.
