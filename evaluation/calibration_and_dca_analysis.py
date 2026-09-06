from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.proportion import proportion_confint


MODEL_FILES = {
    "M1": "M1_Dose_Only_patient_averaged_oof_309.csv",
    "M2-Abl": "M2_Abl_patient_averaged_oof_309.csv",
    "M3-Abl": "M3_Abl_patient_averaged_oof_309.csv",
    "M3": "M3_Oral_GTV_MaskedDose_patient_averaged_oof_309.csv",
}
EXPECTED_COHORT_N = {"Development": 309, "External": 180}
EXPECTED_EVENTS = {"Development": 151, "External": 92}
THRESHOLDS = np.arange(0.01, 1.00, 0.01)
REFERENCE_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
PROBABILITY_EPSILON = 1e-6


def require_env_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Set {name} to an absolute path before running this script.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path: {value!r}")
    return path.resolve()


def optional_env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default.resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"{name} must be an absolute path: {value!r}")
    return path.resolve()


def stable_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(
        np.asarray(probability, dtype=float),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    return np.log(probability / (1.0 - probability))


def calibration_statistics(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    """Joint calibration intercept/slope and CITL with slope fixed at one."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    logits = stable_logit(probability)

    joint = sm.GLM(
        y_true,
        sm.add_constant(logits),
        family=sm.families.Binomial(),
    ).fit()
    joint_ci = np.asarray(joint.conf_int(alpha=0.05), dtype=float)

    citl = sm.GLM(
        y_true,
        np.ones((len(y_true), 1), dtype=float),
        family=sm.families.Binomial(),
        offset=logits,
    ).fit()
    citl_ci = np.asarray(citl.conf_int(alpha=0.05), dtype=float)

    return {
        "calibration_intercept": float(joint.params[0]),
        "calibration_intercept_ci_low": float(joint_ci[0, 0]),
        "calibration_intercept_ci_high": float(joint_ci[0, 1]),
        "calibration_slope": float(joint.params[1]),
        "calibration_slope_ci_low": float(joint_ci[1, 0]),
        "calibration_slope_ci_high": float(joint_ci[1, 1]),
        "calibration_in_the_large": float(citl.params[0]),
        "calibration_in_the_large_ci_low": float(citl_ci[0, 0]),
        "calibration_in_the_large_ci_high": float(citl_ci[0, 1]),
    }


def equal_frequency_calibration_bins(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Create qcut bins and Wilson intervals for observed event proportions."""
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    bins = pd.qcut(probability, q=n_bins, labels=False, duplicates="drop")
    unique_bins = np.unique(np.asarray(bins, dtype=int))
    if len(unique_bins) != n_bins:
        raise RuntimeError(
            f"Expected {n_bins} equal-frequency calibration groups, "
            f"but qcut produced {len(unique_bins)}."
        )

    rows: list[dict[str, float | int]] = []
    for bin_index in unique_bins:
        mask = np.asarray(bins, dtype=int) == int(bin_index)
        n = int(mask.sum())
        events = int(y_true[mask].sum())
        ci_low, ci_high = proportion_confint(
            count=events,
            nobs=n,
            alpha=0.05,
            method="wilson",
        )
        rows.append(
            {
                "bin": int(bin_index) + 1,
                "n": n,
                "mean_predicted": float(probability[mask].mean()),
                "observed_fraction": float(y_true[mask].mean()),
                "observed_ci_low": float(ci_low),
                "observed_ci_high": float(ci_high),
                "min_predicted": float(probability[mask].min()),
                "max_predicted": float(probability[mask].max()),
            }
        )
    return pd.DataFrame(rows)


def decision_curve_rows(
    y_true: np.ndarray,
    probability: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    n = int(len(y_true))
    prevalence = float(y_true.mean())
    rows: list[dict[str, float | int]] = []

    for threshold in np.asarray(thresholds, dtype=float):
        predicted_positive = probability >= threshold
        tp = int(np.logical_and(predicted_positive, y_true == 1).sum())
        fp = int(np.logical_and(predicted_positive, y_true == 0).sum())
        fn = int(np.logical_and(~predicted_positive, y_true == 1).sum())
        tn = int(np.logical_and(~predicted_positive, y_true == 0).sum())
        odds = threshold / (1.0 - threshold)
        model_nb = tp / n - fp / n * odds
        treat_all_nb = prevalence - (1.0 - prevalence) * odds
        rows.append(
            {
                "threshold_probability": float(threshold),
                "n": n,
                "prevalence": prevalence,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "net_benefit_model": float(model_nb),
                "net_benefit_treat_all": float(treat_all_nb),
                "net_benefit_treat_none": 0.0,
                "standardized_net_benefit_model": float(model_nb / prevalence),
                "standardized_net_benefit_treat_all": float(
                    treat_all_nb / prevalence
                ),
                "net_benefit_vs_treat_all": float(model_nb - treat_all_nb),
                "net_benefit_vs_treat_none": float(model_nb),
            }
        )
    return pd.DataFrame(rows)


def _check_prediction_frame(
    frame: pd.DataFrame,
    cohort: str,
    model: str,
) -> pd.DataFrame:
    required = {"patient_key", "y_true", "prob"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"{cohort}/{model} prediction data missing: {sorted(missing)}")

    frame = frame.copy()
    frame["patient_key"] = pd.to_numeric(
        frame["patient_key"], errors="raise"
    ).astype(int)
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="raise").astype(int)
    frame["prob"] = pd.to_numeric(frame["prob"], errors="raise").astype(float)
    if frame["patient_key"].duplicated().any():
        raise RuntimeError(f"{cohort}/{model} contains duplicate patient identifiers")
    if not frame["y_true"].isin([0, 1]).all():
        raise RuntimeError(f"{cohort}/{model} contains non-binary outcomes")
    if not np.isfinite(frame["prob"]).all():
        raise RuntimeError(f"{cohort}/{model} contains non-finite probabilities")
    if not frame["prob"].between(0.0, 1.0, inclusive="both").all():
        raise RuntimeError(f"{cohort}/{model} contains probabilities outside [0, 1]")
    expected_n = EXPECTED_COHORT_N[cohort]
    expected_events = EXPECTED_EVENTS[cohort]
    if len(frame) != expected_n or int(frame["y_true"].sum()) != expected_events:
        raise RuntimeError(
            f"{cohort}/{model} cohort lock failed: "
            f"n/events={len(frame)}/{int(frame['y_true'].sum())}, "
            f"expected={expected_n}/{expected_events}"
        )
    return frame.sort_values("patient_key").reset_index(drop=True)


def load_predictions(
    development_dir: Path,
    external_dir: Path,
) -> dict[tuple[str, str], pd.DataFrame]:
    predictions: dict[tuple[str, str], pd.DataFrame] = {}
    for model, filename in MODEL_FILES.items():
        source = pd.read_csv(development_dir / filename)
        source = source.rename(
            columns={
                "patient_id": "patient_key",
                "oof_probability_mean": "prob",
            }
        )
        frame = source[["patient_key", "y_true", "prob"]].copy()
        predictions[("Development", model)] = _check_prediction_frame(
            frame, "Development", model
        )

    external_source = pd.read_csv(
        external_dir / "external_patient_20model_ensemble_all_models.csv"
    )
    required_external = {"display_name", "patient_id", "y_true", "probability"}
    missing = required_external - set(external_source.columns)
    if missing:
        raise KeyError(f"External prediction table missing: {sorted(missing)}")
    for model in MODEL_FILES:
        source = external_source.loc[
            external_source["display_name"].astype(str).eq(model),
            ["patient_id", "y_true", "probability"],
        ].rename(columns={"patient_id": "patient_key", "probability": "prob"})
        predictions[("External", model)] = _check_prediction_frame(
            source, "External", model
        )

    for cohort in EXPECTED_COHORT_N:
        reference = predictions[(cohort, "M3")][["patient_key", "y_true"]]
        for model in MODEL_FILES:
            candidate = predictions[(cohort, model)][["patient_key", "y_true"]]
            if not candidate.equals(reference):
                raise RuntimeError(
                    f"Patient/outcome alignment differs within {cohort}: M3 versus {model}"
                )
    return predictions


def build_calibration_outputs(
    predictions: dict[tuple[str, str], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, float | int | str]] = []
    bin_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []

    for cohort in ("Development", "External"):
        frame = predictions[(cohort, "M3")].copy()
        y_true = frame["y_true"].to_numpy()
        probability = frame["prob"].to_numpy()
        row: dict[str, float | int | str] = {
            "cohort": cohort,
            "model": "M3",
            "n": int(len(frame)),
            "events": int(y_true.sum()),
            "non_events": int(len(frame) - y_true.sum()),
            "prevalence": float(y_true.mean()),
            "auc_QA": float(roc_auc_score(y_true, probability)),
            "brier_score": float(brier_score_loss(y_true, probability)),
        }
        row.update(calibration_statistics(y_true, probability))
        metric_rows.append(row)

        bins = equal_frequency_calibration_bins(y_true, probability)
        bins.insert(0, "model", "M3")
        bins.insert(0, "cohort", cohort)
        bin_frames.append(bins)

        exported = frame.copy()
        exported.insert(0, "model", "M3")
        exported.insert(0, "cohort", cohort)
        prediction_frames.append(exported)

    return (
        pd.DataFrame(metric_rows),
        pd.concat(bin_frames, ignore_index=True),
        pd.concat(prediction_frames, ignore_index=True),
    )


def build_dca_outputs(
    predictions: dict[tuple[str, str], pd.DataFrame]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dca_frames: list[pd.DataFrame] = []
    qa_rows: list[dict[str, float | int | str]] = []
    for cohort in ("Development", "External"):
        for model in MODEL_FILES:
            frame = predictions[(cohort, model)]
            y_true = frame["y_true"].to_numpy()
            probability = frame["prob"].to_numpy()
            dca = decision_curve_rows(y_true, probability)
            dca.insert(0, "model", model)
            dca.insert(0, "cohort", cohort)
            dca_frames.append(dca)
            qa_rows.append(
                {
                    "cohort": cohort,
                    "model": model,
                    "n": int(len(frame)),
                    "events": int(y_true.sum()),
                    "non_events": int(len(frame) - y_true.sum()),
                    "prevalence": float(y_true.mean()),
                    "unique_patient_ids": int(frame["patient_key"].nunique()),
                    "duplicate_patient_ids": int(frame["patient_key"].duplicated().sum()),
                    "nonfinite_probabilities": int((~np.isfinite(probability)).sum()),
                    "minimum_probability": float(probability.min()),
                    "maximum_probability": float(probability.max()),
                    "mean_probability": float(probability.mean()),
                }
            )

    all_thresholds = pd.concat(dca_frames, ignore_index=True)
    reference = all_thresholds.loc[
        np.isclose(
            all_thresholds["threshold_probability"].to_numpy()[:, None],
            np.asarray(REFERENCE_THRESHOLDS, dtype=float)[None, :],
            atol=1e-12,
            rtol=0.0,
        ).any(axis=1),
        [
            "cohort",
            "model",
            "threshold_probability",
            "net_benefit_model",
            "net_benefit_treat_all",
            "net_benefit_treat_none",
            "net_benefit_vs_treat_all",
            "net_benefit_vs_treat_none",
            "standardized_net_benefit_model",
        ],
    ].reset_index(drop=True)
    return all_thresholds, reference, pd.DataFrame(qa_rows)


def run_analysis() -> dict[str, Path]:
    project_root = require_env_path("NPC_PROJECT_ROOT")
    development_dir = optional_env_path(
        "NPC_DEVELOPMENT_PREDICTIONS_DIR",
        project_root
        / "formal_main_models_no_scalar_4x5_497_BCdev_Aexternal_v2"
        / "aggregate",
    )
    external_dir = optional_env_path(
        "NPC_EXTERNAL_PREDICTIONS_DIR",
        project_root / "external_validation_CenterA_FINAL180_RERUN_v2",
    )
    calibration_dir = optional_env_path(
        "NPC_CALIBRATION_OUTPUT_DIR", project_root / "calibration_FINAL_v3"
    )
    dca_dir = optional_env_path("NPC_DCA_OUTPUT_DIR", project_root / "DCA_FINAL_v1")
    calibration_dir.mkdir(parents=True, exist_ok=True)
    dca_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_predictions(development_dir, external_dir)
    calibration_metrics, calibration_bins, m3_predictions = (
        build_calibration_outputs(predictions)
    )
    dca_all, dca_reference, dca_qa = build_dca_outputs(predictions)

    calibration_path = calibration_dir / "NPC_3DCNN_Calibration_FINAL_v3.xlsx"
    dca_path = dca_dir / "NPC_3DCNN_DCA_FINAL_v1.xlsx"
    with pd.ExcelWriter(calibration_path, engine="openpyxl") as writer:
        calibration_metrics.to_excel(
            writer, sheet_name="Calibration_metrics", index=False
        )
        calibration_bins.to_excel(writer, sheet_name="Calibration_bins", index=False)
        m3_predictions.to_excel(writer, sheet_name="M3_predictions_used", index=False)
    with pd.ExcelWriter(dca_path, engine="openpyxl") as writer:
        dca_all.to_excel(writer, sheet_name="DCA_all_thresholds", index=False)
        dca_reference.to_excel(writer, sheet_name="Reference_thresholds", index=False)
        dca_qa.to_excel(writer, sheet_name="Input_prediction_QA", index=False)

    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_lock": {"assessed": 540, "excluded": 51, "included": 489},
        "development": {"n": 309, "events": 151, "centers": ["B", "C"]},
        "external": {"n": 180, "events": 92, "center": "A"},
        "calibration": {
            "model": "M3",
            "groups": 10,
            "grouping": "pandas qcut equal-frequency groups",
            "observed_interval": "two-sided 95% Wilson interval",
            "logit_clip": [PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON],
            "intercept_and_slope": "binomial GLM of outcome on prediction logit",
            "citl": "binomial GLM intercept with prediction logit as unit-slope offset",
            "external_recalibration": False,
        },
        "decision_curve": {
            "models": list(MODEL_FILES),
            "thresholds": [0.01, 0.99, 0.01],
            "prediction_rule": "probability >= threshold",
            "comparators": ["treat all", "treat none"],
        },
        "outputs": {
            "calibration_workbook": str(calibration_path),
            "dca_workbook": str(dca_path),
        },
        "status": "PASS",
    }
    audit_path = calibration_dir / "calibration_and_dca_analysis_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Calibration workbook: {calibration_path}")
    print(f"DCA workbook: {dca_path}")
    print("[PASS] Calibration and DCA were regenerated from frozen predictions.")
    return {
        "calibration_workbook": calibration_path,
        "dca_workbook": dca_path,
        "audit": audit_path,
    }


if __name__ == "__main__":
    run_analysis()
