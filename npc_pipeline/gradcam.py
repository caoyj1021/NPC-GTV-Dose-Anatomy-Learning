from .common import config, new_output, load_cohort, register
from .predictions import load_aligned
c = config()
load_cohort(c)
load_aligned()
from .artifacts import verify_artifact, lock_artifact
verify_artifact(c['calibration_dir'], 'calibration')
from .cnn_engine import LightweightResNet10_3D
from pathlib import Path
import re
import json
import math
import warnings
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
ROOT = Path(c['result_root'])
MASTER_PATH = Path(c['master'])
CALIBRATION_XLSX = Path(c['calibration_dir']) / 'NPC_3DCNN_Calibration.xlsx'
NPZ_DIR = Path(c['npz_dir'])
FORMAL_MODEL_ROOT = Path(c.get('frozen_sources', {}).get('cnn_root') or c['result_root'])
M3_MODEL_DIRNAME = 'M3_Oral_GTV_MaskedDose'
M3_CHANNELS = ['dose', 'oral', 'gtv', 'dose_oral', 'dose_gtv']
STAGE_B_CKPT_NAME = 'stage_B_outer_train_final_model.pt'
OUTER_VAL_PRED_NAME = 'outer_validation_predictions.csv'
M3_LOCKED_THRESHOLD = float(c['m3_locked_threshold'])
if c.get('cnn_runs', {}).get('M3_Oral_GTV_MaskedDose'):
    M3_LOCKED_THRESHOLD = float(json.loads((Path(c['cnn_runs']['M3_Oral_GTV_MaskedDose']) / 'TRAINING_FROZEN.json').read_text(encoding='utf-8'))['development_youden']['threshold'])
REPEATS = range(1, 5)
FOLDS = range(1, 6)
EXPECTED_DEV_N = 309
EXPECTED_EXT_N = 180
EXPECTED_DEV_OOF_CHECKPOINTS_PER_PATIENT = 4
EXPECTED_EXT_ENSEMBLE_CHECKPOINTS = 20
EXPECTED_SHAPE_ZYX = (64, 112, 80)
BASE_CHANNELS = 16
DROPOUT = 0.2
PROB_QA_TOL = 0.005
CASES_PER_CATEGORY_PER_COHORT = 1
GRADCAM_TARGET_LAYER = 'layer4'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = new_output('GradCAM')
FIG_DIR = OUT_DIR / 'case_figures'
CAM_DIR = OUT_DIR / 'candidate_arrays'
GRADCAM_CASES_CSV = c.get('gradcam_cases_csv', '')
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
CAM_DIR.mkdir(parents=True, exist_ok=True)

def normalize_patient_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == '':
        return None
    if re.fullmatch('[+-]?\\d+(\\.0+)?', s):
        try:
            return str(int(float(s)))
        except Exception:
            pass
    m = re.fullmatch('[Pp](\\d+)', s)
    if m:
        return str(int(m.group(1)))
    return s

def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Required file not found:\n{path}')
    return path

def require_dir(path):
    path = Path(path)
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f'Required directory not found:\n{path}')
    return path

def sigmoid_scalar(x):
    x = float(x)
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)

def build_exact_m3_model():
    return LightweightResNet10_3D(in_channels=len(M3_CHANNELS), base_channels=BASE_CHANNELS, dropout=DROPOUT)

def load_formal_master():
    require_file(MASTER_PATH)
    master = pd.read_excel(MASTER_PATH, sheet_name=c.get('master_sheet', 0))
    required = {'patient_id', 'model_center', 'severe_mucositis', 'exclude_reason'}
    missing = required - set(master.columns)
    if missing:
        raise KeyError('Frozen Master missing columns: ' + ', '.join(sorted(missing)))
    keep_cols = [c for c in master.columns if c in {'patient_id', 'model_center', 'severe_mucositis', 'exclude_reason', 'original_center', 'ct_path', 'dose_path', 'oral_cavity_mask_path', 'gtv_mask_path'}]
    df = master[keep_cols].copy()
    df['patient_key'] = df['patient_id'].map(normalize_patient_id)
    df['model_center'] = df['model_center'].astype(str).str.strip().str.upper()
    y_num = pd.to_numeric(df['severe_mucositis'], errors='coerce')
    exclude_text = df['exclude_reason'].fillna('').astype(str).str.strip()
    formal_mask = exclude_text.eq('') & df['patient_key'].notna() & df['model_center'].isin(['A', 'B', 'C']) & y_num.isin([0, 1])
    formal = df.loc[formal_mask].copy()
    formal['y_true'] = pd.to_numeric(formal['severe_mucositis'], errors='raise').astype(int)
    formal['cohort'] = np.where(formal['model_center'].eq('A'), 'External', 'Development')
    if formal['patient_key'].duplicated().any():
        dup = formal.loc[formal['patient_key'].duplicated(False), ['patient_id', 'patient_key', 'model_center']]
        raise ValueError('Duplicate patient IDs in formal cohort:\n' + dup.to_string(index=False))
    dev_n = int((formal['cohort'] == 'Development').sum())
    ext_n = int((formal['cohort'] == 'External').sum())
    if dev_n != EXPECTED_DEV_N or ext_n != EXPECTED_EXT_N:
        raise ValueError(f'[COHORT QA FAIL] Development={dev_n}, External={ext_n}; expected {EXPECTED_DEV_N}/{EXPECTED_EXT_N}.')
    return formal

def load_locked_m3_predictions():
    require_file(CALIBRATION_XLSX)
    pred = pd.read_excel(CALIBRATION_XLSX, sheet_name='M3_predictions_used')
    required = {'cohort', 'model', 'patient_key', 'y_true', 'prob'}
    missing = required - set(pred.columns)
    if missing:
        raise KeyError('M3_predictions_used missing columns: ' + ', '.join(sorted(missing)))
    pred = pred[pred['model'].astype(str).str.upper().eq('M3')].copy()
    pred['patient_key'] = pred['patient_key'].map(normalize_patient_id)
    pred['y_true'] = pd.to_numeric(pred['y_true'], errors='raise').astype(int)
    pred['prob'] = pd.to_numeric(pred['prob'], errors='raise').astype(float)
    return pred

def build_locked_case_table():
    formal = load_formal_master()
    pred = load_locked_m3_predictions()
    merged = formal.merge(pred[['cohort', 'patient_key', 'y_true', 'prob']], on=['cohort', 'patient_key', 'y_true'], how='inner', validate='one_to_one')
    dev_n = int((merged['cohort'] == 'Development').sum())
    ext_n = int((merged['cohort'] == 'External').sum())
    if dev_n != EXPECTED_DEV_N or ext_n != EXPECTED_EXT_N:
        raise ValueError(f'[PREDICTION MERGE QA FAIL] Development={dev_n}, External={ext_n}; expected {EXPECTED_DEV_N}/{EXPECTED_EXT_N}.')
    merged['threshold'] = M3_LOCKED_THRESHOLD
    merged['y_pred'] = (merged['prob'] >= M3_LOCKED_THRESHOLD).astype(int)

    def category(row):
        y = int(row['y_true'])
        p = int(row['y_pred'])
        if y == 1 and p == 1:
            return 'TP'
        if y == 0 and p == 0:
            return 'TN'
        if y == 0 and p == 1:
            return 'FP'
        if y == 1 and p == 0:
            return 'FN'
        raise RuntimeError('Unexpected binary state.')
    merged['category'] = merged.apply(category, axis=1)
    return merged

def rank_representative_cases(sub, category):
    d = sub.copy()
    if len(d) == 0:
        return d
    if category in ['TP', 'TN']:
        median_prob = float(d['prob'].median())
        d['representative_score'] = (d['prob'] - median_prob).abs()
        d['representative_rule'] = f'closest_to_category_median_{median_prob:.6f}'
    else:
        d['representative_score'] = (d['prob'] - M3_LOCKED_THRESHOLD).abs()
        d['representative_rule'] = f'closest_to_locked_threshold_{M3_LOCKED_THRESHOLD:.10f}'
    d = d.sort_values(['representative_score', 'patient_key'], ascending=[True, True]).reset_index(drop=True)
    d['representative_rank'] = np.arange(len(d)) + 1
    return d

def select_representative_cases(case_table):
    selected = []
    rankings = []
    for cohort in ['Development', 'External']:
        for category in ['TP', 'TN', 'FP', 'FN']:
            sub = case_table[(case_table['cohort'] == cohort) & (case_table['category'] == category)].copy()
            ranked = rank_representative_cases(sub, category)
            if len(ranked) == 0:
                continue
            rankings.append(ranked)
            selected.append(ranked.head(CASES_PER_CATEGORY_PER_COHORT).copy())
    return (pd.concat(selected, ignore_index=True), pd.concat(rankings, ignore_index=True))

def find_npz(patient_key):
    patient_key = normalize_patient_id(patient_key)
    direct = NPZ_DIR / f'{patient_key}.npz'
    if direct.exists():
        return direct
    if str(patient_key).isdigit():
        for name in [f'{int(patient_key):03d}.npz', f'P{int(patient_key):03d}.npz']:
            p = NPZ_DIR / name
            if p.exists():
                return p
    hits = []
    for p in NPZ_DIR.glob('*.npz'):
        if normalize_patient_id(p.stem) == patient_key:
            hits.append(p)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise RuntimeError(f'Multiple NPZ files found for patient {patient_key}:\n' + '\n'.join((str(x) for x in hits)))
    raise FileNotFoundError(f'NPZ not found for patient {patient_key} in {NPZ_DIR}')

def load_exact_m3_input(npz_path):
    """
    Reproduce the formal Dataset input exactly.
    NPZ label/cohort fields are intentionally ignored.
    """
    npz_path = require_file(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        required = ['ct', 'dose', 'oral', 'gtv', 'patient_id', 'patch_direction', 'dose_normalization_gy', 'dose_scale_to_gy', 'dose_normalized_max']
        for key in required:
            if key not in data.files:
                raise KeyError(f'{npz_path.name} missing required array: {key}')
        embedded_id = np.asarray(data['patient_id'])
        if embedded_id.size != 1 or not np.isfinite(embedded_id).all() or float(embedded_id.item()) != int(embedded_id.item()):
            raise RuntimeError('Grad-CAM NPZ patient_id must be one finite integer')
        if str(int(embedded_id.item())) != normalize_patient_id(npz_path.stem):
            raise RuntimeError('Grad-CAM NPZ patient_id does not match its filename')
        ct = np.asarray(data['ct'], dtype=np.float32)
        dose = np.asarray(data['dose'], dtype=np.float32)
        oral_raw = np.asarray(data['oral'], dtype=np.float32)
        gtv_raw = np.asarray(data['gtv'], dtype=np.float32)
        patch_direction = np.asarray(data['patch_direction'], dtype=np.float64).reshape(3, 3)
        dose_normalization_gy = float(np.asarray(data['dose_normalization_gy']).reshape(-1)[0])
        dose_scale_to_gy = float(np.asarray(data['dose_scale_to_gy']).reshape(-1)[0])
        dose_normalized_max = float(np.asarray(data['dose_normalized_max']).reshape(-1)[0])
    for name, arr in [('ct', ct), ('dose', dose), ('oral', oral_raw), ('gtv', gtv_raw)]:
        if tuple(arr.shape) != EXPECTED_SHAPE_ZYX:
            raise RuntimeError(f'{npz_path.name}: {name} shape={arr.shape}, expected={EXPECTED_SHAPE_ZYX}')
        if not np.isfinite(arr).all():
            raise RuntimeError(f'{npz_path.name}: {name} contains NaN/Inf')
    oral = (oral_raw > 0.5).astype(np.float32)
    gtv = (gtv_raw > 0.5).astype(np.float32)
    if int(oral.sum()) <= 0:
        raise RuntimeError(f'{npz_path.name}: oral mask is empty')
    if int(gtv.sum()) <= 0:
        raise RuntimeError(f'{npz_path.name}: gtv mask is empty')
    dose_oral = (dose * oral).astype(np.float32, copy=False)
    dose_gtv = (dose * gtv).astype(np.float32, copy=False)
    arrays = {'dose': dose, 'oral': oral, 'gtv': gtv, 'dose_oral': dose_oral, 'dose_gtv': dose_gtv}
    image = np.stack([arrays[ch] for ch in M3_CHANNELS], axis=0).astype(np.float32, copy=False)
    expected = (len(M3_CHANNELS), *EXPECTED_SHAPE_ZYX)
    if tuple(image.shape) != expected:
        raise RuntimeError(f'{npz_path.name}: M3 input shape={image.shape}, expected={expected}')
    return {'input_tensor': np.ascontiguousarray(image), 'ct': ct, 'dose': dose, 'oral': oral, 'gtv': gtv, 'dose_oral': dose_oral, 'dose_gtv': dose_gtv, 'patch_direction': patch_direction, 'dose_normalization_gy': dose_normalization_gy, 'dose_scale_to_gy': dose_scale_to_gy, 'dose_normalized_max': dose_normalized_max}

def task_dir(repeat_number, fold_number):
    return FORMAL_MODEL_ROOT / f'repeat_{repeat_number:02d}' / f'fold_{fold_number:02d}' / M3_MODEL_DIRNAME

def build_original_checkpoint_table():
    rows = []
    for repeat_number in REPEATS:
        for fold_number in FOLDS:
            td = task_dir(repeat_number, fold_number)
            ckpt = td / STAGE_B_CKPT_NAME
            pred_csv = td / OUTER_VAL_PRED_NAME
            if not ckpt.exists():
                raise FileNotFoundError(f'Missing M3 Stage-B checkpoint:\n{ckpt}')
            if not pred_csv.exists():
                raise FileNotFoundError(f'Missing M3 outer-validation prediction file:\n{pred_csv}')
            rows.append({'repeat': repeat_number, 'fold': fold_number, 'task_dir': str(td), 'checkpoint_path': str(ckpt), 'outer_validation_predictions_csv': str(pred_csv)})
    df = pd.DataFrame(rows)
    if len(df) != EXPECTED_EXT_ENSEMBLE_CHECKPOINTS:
        raise RuntimeError(f'Expected 20 M3 Stage-B checkpoints, found {len(df)}.')
    return df

def build_all_20_checkpoint_table():
    run = c.get('cnn_runs', {}).get('M3_Oral_GTV_MaskedDose')
    if run:
        folder = Path(run)
        frozen = json.loads((folder / 'TRAINING_FROZEN.json').read_text(encoding='utf-8'))
        rows = []
        for r in frozen['records']:
            rows.append({'repeat': r['repeat'], 'fold': r['fold'], 'task_dir': str((folder / r['checkpoint']).parent), 'checkpoint_path': str(folder / r['checkpoint']), 'outer_validation_predictions_csv': str(folder / r['oof_csv'])})
        return pd.DataFrame(rows)
    return build_original_checkpoint_table()

def load_checkpoint_exact(ckpt_path, device):
    ckpt_path = require_file(ckpt_path)
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f'Unexpected checkpoint object type: {type(payload)}')
    if 'model_state' not in payload:
        raise KeyError(f"Checkpoint missing formal key 'model_state':\n{ckpt_path}\nAvailable keys: {list(payload.keys())}")
    model_name = str(payload.get('model_name', ''))
    if model_name and model_name != M3_MODEL_DIRNAME:
        raise RuntimeError(f'Wrong model checkpoint: model_name={model_name}, expected={M3_MODEL_DIRNAME}')
    channels = payload.get('channels', None)
    if channels is not None:
        channels = list(channels)
        if channels != M3_CHANNELS:
            raise RuntimeError(f'Checkpoint channel mismatch:\ncheckpoint={channels}\nexpected={M3_CHANNELS}')
    model = build_exact_m3_model().to(device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()
    return (model, payload)

def development_oof_checkpoint_rows(patient_key, all_checkpoint_df):
    """
    Identify exactly the 4 Stage-B checkpoints where the Development patient
    was in outer_validation (one fold per repeat).
    """
    pid = int(patient_key)
    hits = []
    for _, row in all_checkpoint_df.iterrows():
        pred_csv = Path(row['outer_validation_predictions_csv'])
        pred = pd.read_csv(pred_csv, encoding='utf-8-sig')
        if 'patient_id' not in pred.columns:
            raise KeyError(f'{pred_csv} missing patient_id column.')
        ids = pd.to_numeric(pred['patient_id'], errors='coerce')
        if (ids == pid).any():
            hits.append(row.to_dict())
    hit_df = pd.DataFrame(hits)
    if len(hit_df) != EXPECTED_DEV_OOF_CHECKPOINTS_PER_PATIENT:
        raise RuntimeError(f'Development patient {patient_key}: found {len(hit_df)} OOF checkpoints, expected 4.')
    repeat_counts = hit_df['repeat'].value_counts().sort_index()
    if set(repeat_counts.index.tolist()) != {1, 2, 3, 4}:
        raise RuntimeError(f'Development patient {patient_key}: OOF checkpoint repeats are {repeat_counts.to_dict()}, expected one each for repeats 1-4.')
    if not (repeat_counts == 1).all():
        raise RuntimeError(f'Development patient {patient_key}: duplicate OOF fold within repeat.')
    return hit_df.sort_values(['repeat', 'fold']).reset_index(drop=True)

def get_module(model, name):
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(f"Grad-CAM target layer '{name}' not found.")
    return modules[name]

class BinaryGradCAM3D:
    """
    Grad-CAM targeting the SEVERE-MUCOSITIS logit.

    Formal model output is a single scalar logit z:
        P(severe) = sigmoid(z)

    Therefore the positive/severe target score is z itself.
    """

    def __init__(self, model, target_layer_name='layer4'):
        self.model = model
        self.target_layer_name = target_layer_name
        self.activations = None
        self.gradients = None
        target_layer = get_module(model, target_layer_name)

        def forward_hook(module, inputs, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        self.forward_handle = target_layer.register_forward_hook(forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(backward_hook)

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

    def __call__(self, x):
        self.model.zero_grad(set_to_none=True)
        logit = self.model(x)
        if logit.ndim == 0:
            logit = logit.unsqueeze(0)
        if logit.numel() != 1:
            raise RuntimeError(f'Expected one scalar logit for one patient, got shape {tuple(logit.shape)}')
        score = logit.sum()
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError('Grad-CAM hooks did not capture activations/gradients.')
        A = self.activations
        G = self.gradients
        if A.ndim != 5 or G.ndim != 5:
            raise RuntimeError(f'Expected 5D feature maps, got A={A.shape}, G={G.shape}')
        weights = G.mean(dim=(2, 3, 4), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode='trilinear', align_corners=False)
        cam = cam[0, 0].detach().float().cpu().numpy()
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max > cam_min:
            cam = ((cam - cam_min) / (cam_max - cam_min)).astype(np.float32)
        else:
            cam = np.zeros_like(cam, dtype=np.float32)
        logit_value = float(logit.detach().float().cpu().reshape(-1)[0].item())
        prob_value = sigmoid_scalar(logit_value)
        return {'cam': cam, 'logit': logit_value, 'probability': prob_value}

def safe_mean(arr, mask):
    mask = np.asarray(mask).astype(bool)
    if mask.sum() == 0:
        return np.nan
    return float(np.asarray(arr, dtype=float)[mask].mean())

def summarize_cam_spatially(cam, dose, oral, gtv):
    cam = np.asarray(cam, dtype=np.float32)
    dose = np.asarray(dose, dtype=np.float32)
    oral = np.asarray(oral, dtype=np.float32) > 0.5
    gtv = np.asarray(gtv, dtype=np.float32) > 0.5
    positive_cam = cam[cam > 0]
    if positive_cam.size > 0:
        threshold_90 = float(np.quantile(positive_cam, 0.9))
    else:
        threshold_90 = 1.0
    high_cam = cam >= threshold_90
    high_cam_n = int(high_cam.sum())

    def fraction_of_high_cam_in(mask):
        if high_cam_n <= 0:
            return np.nan
        return float(np.logical_and(high_cam, mask).sum() / high_cam_n)
    oral_cam_mean = safe_mean(cam, oral)
    gtv_cam_mean = safe_mean(cam, gtv)
    outside_oral_cam_mean = safe_mean(cam, ~oral)
    outside_gtv_cam_mean = safe_mean(cam, ~gtv)
    return {'cam_mean_oral': oral_cam_mean, 'cam_mean_outside_oral': outside_oral_cam_mean, 'cam_oral_to_outside_ratio': oral_cam_mean / outside_oral_cam_mean if np.isfinite(oral_cam_mean) and np.isfinite(outside_oral_cam_mean) and (outside_oral_cam_mean > 0) else np.nan, 'cam_mean_gtv': gtv_cam_mean, 'cam_mean_outside_gtv': outside_gtv_cam_mean, 'cam_gtv_to_outside_ratio': gtv_cam_mean / outside_gtv_cam_mean if np.isfinite(gtv_cam_mean) and np.isfinite(outside_gtv_cam_mean) and (outside_gtv_cam_mean > 0) else np.nan, 'top10pct_cam_voxels': high_cam_n, 'top10pct_cam_fraction_in_oral': fraction_of_high_cam_in(oral), 'top10pct_cam_fraction_in_gtv': fraction_of_high_cam_in(gtv), 'dose_mean_top10pct_cam': safe_mean(dose, high_cam), 'dose_mean_oral': safe_mean(dose, oral), 'dose_mean_gtv': safe_mean(dose, gtv)}

def robust_window(arr, p_low=1, p_high=99):
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(arr, p_low))
    hi = float(np.percentile(arr, p_high))
    if hi <= lo:
        lo = float(arr.min())
        hi = float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)

def slices_at_zyx(arr, zyx):
    z, y, x = [int(v) for v in zyx]
    return {'Axial': arr[z, :, :], 'Coronal': arr[:, y, :], 'Sagittal': arr[:, :, x]}

def add_mask_contour(ax, mask2d, linestyle='-'):
    if mask2d is None:
        return
    mask2d = np.asarray(mask2d, dtype=np.float32)
    if mask2d.max() <= 0:
        return
    ax.contour(mask2d, levels=[0.5], linewidths=0.9, linestyles=linestyle)

def render_case(case_row, payload, ensemble_cam, cam_std, reproduced_probability, checkpoint_count, out_png, out_pdf):
    ct = robust_window(payload['ct'])
    dose = robust_window(payload['dose'])
    oral = payload['oral']
    gtv = payload['gtv']
    max_zyx = np.unravel_index(int(np.argmax(ensemble_cam)), ensemble_cam.shape)
    ct_s = slices_at_zyx(ct, max_zyx)
    dose_s = slices_at_zyx(dose, max_zyx)
    cam_s = slices_at_zyx(ensemble_cam, max_zyx)
    oral_s = slices_at_zyx(oral, max_zyx)
    gtv_s = slices_at_zyx(gtv, max_zyx)
    plane_names = ['Axial', 'Coronal', 'Sagittal']
    fig, axes = plt.subplots(3, 4, figsize=(10.8, 7.8))
    for i, plane in enumerate(plane_names):
        ax = axes[i, 0]
        ax.imshow(ct_s[plane], cmap='gray', interpolation='nearest')
        if i == 0:
            ax.set_title('Planning CT\n(anatomical reference)')
        ax.set_ylabel(plane)
        ax.axis('off')
        ax = axes[i, 1]
        ax.imshow(dose_s[plane], cmap='inferno', interpolation='nearest')
        add_mask_contour(ax, oral_s[plane], linestyle='-')
        add_mask_contour(ax, gtv_s[plane], linestyle='--')
        if i == 0:
            ax.set_title('Dose + oral/GTV contours')
        ax.axis('off')
        ax = axes[i, 2]
        ax.imshow(ct_s[plane], cmap='gray', interpolation='nearest')
        ax.imshow(cam_s[plane], cmap='jet', interpolation='nearest', alpha=0.45)
        add_mask_contour(ax, oral_s[plane], linestyle='-')
        add_mask_contour(ax, gtv_s[plane], linestyle='--')
        if i == 0:
            ax.set_title('M3 Grad-CAM\n(severe-mucositis logit)')
        ax.axis('off')
        ax = axes[i, 3]
        ax.imshow(ct_s[plane], cmap='gray', interpolation='nearest')
        ax.imshow(dose_s[plane], cmap='inferno', interpolation='nearest', alpha=0.28)
        ax.imshow(cam_s[plane], cmap='jet', interpolation='nearest', alpha=0.38)
        add_mask_contour(ax, oral_s[plane], linestyle='-')
        add_mask_contour(ax, gtv_s[plane], linestyle='--')
        if i == 0:
            ax.set_title('Dose + anatomy + Grad-CAM')
        ax.axis('off')
    pid = str(case_row['patient_key'])
    locked_prob = float(case_row['prob'])
    fig.suptitle(f"{case_row['cohort']} | {case_row['category']} | patient {pid} | true={int(case_row['y_true'])} | pred={int(case_row['y_pred'])} | locked p={locked_prob:.3f} | reproduced p={reproduced_probability:.3f} | n models={checkpoint_count}", fontsize=10.5)
    fig.text(0.5, 0.012, 'Solid contour: oral cavity | Dashed contour: GTV | CT is displayed only as anatomical reference and was not an M3 input.', ha='center', fontsize=8.3)
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(out_png, dpi=400, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)
    return {'cam_max_z': int(max_zyx[0]), 'cam_max_y': int(max_zyx[1]), 'cam_max_x': int(max_zyx[2]), 'ensemble_cam_mean': float(ensemble_cam.mean()), 'ensemble_cam_max': float(ensemble_cam.max()), 'checkpoint_cam_std_mean': float(cam_std.mean())}
print('=' * 96)
print('NPC 3D CNN M3 Layer4 Grad-CAM')
print('=' * 96)
print(f'Device: {DEVICE}')
require_dir(NPZ_DIR)
require_dir(FORMAL_MODEL_ROOT)
case_table = build_locked_case_table()
print('\nLOCKED M3 CASE COUNTS')
print(pd.crosstab(case_table['cohort'], case_table['category']).to_string())
selected_df, ranking_df = select_representative_cases(case_table)
if GRADCAM_CASES_CSV:
    case_selection_path = Path(GRADCAM_CASES_CSV).expanduser().resolve()
    require_file(case_selection_path)
    requested = pd.read_csv(case_selection_path)
    if 'patient_id' in requested.columns and 'patient_key' not in requested.columns:
        requested = requested.rename(columns={'patient_id': 'patient_key'})
    missing = {'cohort', 'patient_key'} - set(requested.columns)
    if missing:
        raise KeyError(f'Grad-CAM case-selection CSV missing: {sorted(missing)}')
    requested['patient_key'] = requested['patient_key'].map(normalize_patient_id)
    selected_df = case_table.merge(requested, on=['cohort', 'patient_key'], how='inner', validate='one_to_one', suffixes=('', '_requested'))
    if len(selected_df) != len(requested):
        raise RuntimeError('Not every requested Grad-CAM case matched the final locked cohort.')
    if 'category_requested' in selected_df.columns:
        if not selected_df['category'].eq(selected_df['category_requested']).all():
            raise RuntimeError('Requested Grad-CAM category disagrees with the locked prediction.')
    if 'representative_rank' not in selected_df.columns:
        selected_df['representative_rank'] = 1
    if 'representative_rule' not in selected_df.columns:
        selected_df['representative_rule'] = 'configured_case_selection_csv'
    if 'representative_score' not in selected_df.columns:
        selected_df['representative_score'] = np.nan
print('\nSELECTED REPRESENTATIVE CASES')
print(selected_df[['cohort', 'category', 'patient_key', 'y_true', 'prob', 'y_pred', 'representative_rank', 'representative_rule', 'representative_score']].to_string(index=False))
case_table.to_csv(OUT_DIR / 'M3_locked_case_table.csv', index=False)
selected_df.to_csv(OUT_DIR / 'GradCAM_selected_cases.csv', index=False)
ranking_df.to_csv(OUT_DIR / 'GradCAM_case_rankings.csv', index=False)
figure4_ranking = selected_df.copy()
figure4_ranking['locked_probability'] = figure4_ranking['prob']
if 'primary_rank' not in figure4_ranking.columns:
    figure4_ranking['primary_rank'] = figure4_ranking.get('representative_rank', 1)
if 'illustrative_rank' not in figure4_ranking.columns:
    figure4_ranking['illustrative_rank'] = figure4_ranking.get('representative_rank', 1)
figure4_ranking.to_csv(OUT_DIR / 'GradCAM_display_case_ranking.csv', index=False)
all_ckpt_df = build_all_20_checkpoint_table()
all_ckpt_df.to_csv(OUT_DIR / 'M3_all_20_stageB_checkpoints.csv', index=False)
print(f'\n[CHECKPOINT PASS] Found exactly {len(all_ckpt_df)} formal M3 Stage-B checkpoints.')
model_test = build_exact_m3_model()
parameter_count = sum((p.numel() for p in model_test.parameters()))
print(f'[MODEL PASS] LightweightResNet10_3D | in_channels={len(M3_CHANNELS)} | parameters={parameter_count:,} | target_layer={GRADCAM_TARGET_LAYER}')
del model_test
checkpoint_use_rows = []
render_rows = []
for case_index, case_row in selected_df.iterrows():
    pid = normalize_patient_id(case_row['patient_key'])
    cohort = str(case_row['cohort'])
    category = str(case_row['category'])
    print('\n' + '-' * 96)
    print(f'CASE {case_index + 1}/{len(selected_df)} | {cohort} | {category} | patient {pid}')
    npz_path = find_npz(pid)
    payload = load_exact_m3_input(npz_path)
    x = torch.from_numpy(payload['input_tensor'][None]).float().to(DEVICE)
    if cohort == 'Development':
        use_ckpts = development_oof_checkpoint_rows(pid, all_ckpt_df)
        expected_ckpt_n = 4
        ensemble_type = '4-repeat patient-specific OOF ensemble'
    elif cohort == 'External':
        use_ckpts = all_ckpt_df.copy()
        expected_ckpt_n = 20
        ensemble_type = '20-model external ensemble'
    else:
        raise RuntimeError(f'Unknown cohort: {cohort}')
    if len(use_ckpts) != expected_ckpt_n:
        raise RuntimeError(f'{pid}: checkpoint count={len(use_ckpts)}, expected={expected_ckpt_n}')
    cams = []
    probs = []
    logits = []
    for _, ckpt_row in use_ckpts.iterrows():
        ckpt_path = Path(ckpt_row['checkpoint_path'])
        model, ckpt_payload = load_checkpoint_exact(ckpt_path, DEVICE)
        gradcam = BinaryGradCAM3D(model, target_layer_name=GRADCAM_TARGET_LAYER)
        out = gradcam(x)
        gradcam.close()
        cams.append(out['cam'])
        probs.append(out['probability'])
        logits.append(out['logit'])
        checkpoint_use_rows.append({'patient_key': pid, 'cohort': cohort, 'category': category, 'repeat': int(ckpt_row['repeat']), 'fold': int(ckpt_row['fold']), 'checkpoint_path': str(ckpt_path), 'checkpoint_probability': float(out['probability']), 'checkpoint_logit': float(out['logit']), 'selected_epoch': ckpt_payload.get('selected_epoch', np.nan), 'training_seed': ckpt_payload.get('training_seed', np.nan), 'ensemble_type': ensemble_type})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    cams_np = np.stack(cams, axis=0).astype(np.float32)
    ensemble_cam = cams_np.mean(axis=0).astype(np.float32)
    cam_std = cams_np.std(axis=0, ddof=0).astype(np.float32)
    reproduced_probability = float(np.mean(probs))
    locked_probability = float(case_row['prob'])
    prob_abs_diff = abs(reproduced_probability - locked_probability)
    if prob_abs_diff > PROB_QA_TOL:
        raise RuntimeError(f'[PROBABILITY QA FAIL] patient {pid}\nlocked probability     = {locked_probability:.10f}\nreproduced probability = {reproduced_probability:.10f}\nabsolute difference     = {prob_abs_diff:.10f}\ntolerance               = {PROB_QA_TOL:.10f}\nThis usually indicates the wrong checkpoint root or wrong input/model definition.')
    spatial = summarize_cam_spatially(ensemble_cam, payload['dose'], payload['oral'], payload['gtv'])
    cam_npz_path = CAM_DIR / f'{cohort}_{category}_{pid}_screening_v1.npz'
    np.savez_compressed(cam_npz_path, patient_id=str(pid), cohort=cohort, category=category, y_true=int(case_row['y_true']), y_pred=int(case_row['y_pred']), locked_probability=locked_probability, reproduced_probability=reproduced_probability, checkpoint_count=len(use_ckpts), ensemble_cam=ensemble_cam, layer4_mean_cam=ensemble_cam, checkpoint_cam_std=cam_std, patch_direction=payload['patch_direction'], dose_normalization_gy=payload['dose_normalization_gy'], dose_scale_to_gy=payload['dose_scale_to_gy'], dose_normalized_max=payload['dose_normalized_max'], dose=payload['dose'], oral=payload['oral'], gtv=payload['gtv'], ct=payload['ct'])
    fig_base = f'{cohort}_{category}_{pid}'
    fig_png = FIG_DIR / f'{fig_base}.png'
    fig_pdf = FIG_DIR / f'{fig_base}.pdf'
    visual_stats = render_case(case_row, payload, ensemble_cam, cam_std, reproduced_probability, len(use_ckpts), fig_png, fig_pdf)
    render_row = {'patient_key': pid, 'cohort': cohort, 'category': category, 'y_true': int(case_row['y_true']), 'y_pred': int(case_row['y_pred']), 'locked_probability': locked_probability, 'reproduced_probability': reproduced_probability, 'probability_abs_difference': prob_abs_diff, 'ensemble_type': ensemble_type, 'checkpoint_count': len(use_ckpts), 'npz_path': str(npz_path), 'cam_npz_path': str(cam_npz_path), 'figure_png': str(fig_png), 'figure_pdf': str(fig_pdf), **visual_stats, **spatial}
    render_rows.append(render_row)
    print(f'[PASS] {pid} | {ensemble_type} | locked p={locked_probability:.6f} | reproduced p={reproduced_probability:.6f} | |Δp|={prob_abs_diff:.6f}')
checkpoint_use_df = pd.DataFrame(checkpoint_use_rows)
render_df = pd.DataFrame(render_rows)
checkpoint_use_df.to_csv(OUT_DIR / 'GradCAM_checkpoint_use_by_case.csv', index=False)
render_df.to_csv(OUT_DIR / 'GradCAM_render_summary.csv', index=False)
excel_path = OUT_DIR / 'NPC_3DCNN_GradCAM.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    case_table.to_excel(writer, sheet_name='Locked_case_table', index=False)
    selected_df.to_excel(writer, sheet_name='Selected_cases', index=False)
    ranking_df.to_excel(writer, sheet_name='Case_rankings', index=False)
    all_ckpt_df.to_excel(writer, sheet_name='All20_M3_checkpoints', index=False)
    checkpoint_use_df.to_excel(writer, sheet_name='Checkpoint_use_by_case', index=False)
    render_df.to_excel(writer, sheet_name='GradCAM_summary', index=False)
audit = {'analysis': 'M3 Grad-CAM / spatial interpretation', 'implementation': 'Layer4 Grad-CAM', 'master': str(MASTER_PATH), 'calibration_predictions': str(CALIBRATION_XLSX), 'npz_dir': str(NPZ_DIR), 'formal_model_root': str(FORMAL_MODEL_ROOT), 'm3_model_dirname': M3_MODEL_DIRNAME, 'm3_channels': M3_CHANNELS, 'derived_channels': {'dose_oral': 'dose * (oral > 0.5)', 'dose_gtv': 'dose * (gtv > 0.5)'}, 'architecture': {'name': 'LightweightResNet10_3D', 'base_channels': BASE_CHANNELS, 'dropout': DROPOUT, 'output': 'single binary logit', 'probability': 'sigmoid(logit)', 'gradcam_target_layer': GRADCAM_TARGET_LAYER}, 'checkpoint': {'filename': STAGE_B_CKPT_NAME, 'state_key': 'model_state', 'formal_m3_checkpoints_expected': 20}, 'interpretation_ensemble': {'Development': 'patient-specific 4 OOF Stage-B checkpoints only; one held-out fold per repeat', 'External': 'all 20 Stage-B checkpoints; matches final external ensemble'}, 'locked_threshold': M3_LOCKED_THRESHOLD, 'probability_qa_tolerance': PROB_QA_TOL, 'npz_label_read': False, 'npz_cohort_read': False, 'retraining': False, 'threshold_tuning': False, 'model_updating': False, 'status': 'PASS'}
with open(OUT_DIR / 'GradCAM_AUDIT.json', 'w', encoding='utf-8') as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
print('\n' + '=' * 96)
print('[FINAL PASS] Grad-CAM completed.')
print(f'Output folder:\n{OUT_DIR}')
print('\nMain outputs:')
print('  1) NPC_3DCNN_GradCAM.xlsx')
print('  2) GradCAM_selected_cases.csv')
print('  3) GradCAM_case_rankings.csv')
print('  4) M3_all_20_stageB_checkpoints.csv')
print('  5) GradCAM_checkpoint_use_by_case.csv')
print('  6) GradCAM_render_summary.csv')
print('  7) candidate_arrays\\*.npz')
print('  8) case_figures\\*.png / *.pdf')
print('  9) GradCAM_AUDIT.json')
print('=' * 96)
lock_artifact(OUT_DIR, 'gradcam', [Path(c['calibration_dir']) / 'ARTIFACT_LOCK.json', *[Path(x) for x in all_ckpt_df.checkpoint_path], *[Path(x) for x in all_ckpt_df.outer_validation_predictions_csv], *[find_npz(x) for x in selected_df.patient_key]])
register('gradcam_dir', str(OUT_DIR))
