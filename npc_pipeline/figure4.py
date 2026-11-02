from .common import config, new_output, load_cohort, register
from .predictions import load_aligned
c = config()
load_cohort(c)
load_aligned()
from .artifacts import verify_artifact
verify_artifact(c['gradcam_dir'], 'gradcam')
from pathlib import Path
import os
import re
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
ROOT = Path(c['result_root'])
SCREEN_DIR = Path(c['gradcam_dir'])
SCREEN_ARRAY_DIR = SCREEN_DIR / 'candidate_arrays'
RANKING_CSV = SCREEN_DIR / 'GradCAM_display_case_ranking.csv'
ORIGINAL_NPZ_DIR = Path(c['npz_dir'])
OUT_DIR = new_output('Figure4')
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVELOPMENT_CASE_ID = str(c['figure4_cases']['Development'])
EXTERNAL_CASE_ID = str(c['figure4_cases']['External'])
if not DEVELOPMENT_CASE_ID.isdigit() or not EXTERNAL_CASE_ID.isdigit():
    raise RuntimeError('Set figure4_cases.Development and figure4_cases.External in config.local.json to the approved integer patient identifiers.')
MAIN_CASES = [('Development', DEVELOPMENT_CASE_ID, 'Development cohort'), ('External', EXTERNAL_CASE_ID, 'Independent external validation cohort')]
PLANE = 'Sagittal'
NON_AIR_THRESHOLD = -0.95
ORAL_COLOR = '#00A651'
GTV_COLOR = '#00A6D6'
CAM_CMAP = 'jet'
CAM_ALPHA = 0.5
DOSE_CMAP = 'inferno'
DOSE_ALPHA = 0.48

def normalize_patient_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if re.fullmatch('[+-]?\\d+(\\.0+)?', s):
        return str(int(float(s)))
    m = re.fullmatch('[Pp](\\d+)', s)
    if m:
        return str(int(m.group(1)))
    return s

def require_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'Required file not found:\n{path}')
    return path

def direction_to_lps_axis_transform(direction):
    D = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    if not np.allclose(D.T @ D, np.eye(3), atol=0.0001):
        raise RuntimeError(f'patch_direction is not orthonormal:\n{D}')
    permutation_xyz = []
    flip_xyz = []
    used = set()
    for physical_axis in range(3):
        row = np.abs(D[physical_axis, :])
        current_axis = int(np.argmax(row))
        if current_axis in used:
            raise RuntimeError(f'Invalid direction mapping:\n{D}')
        if row[current_axis] < 0.95:
            raise RuntimeError('Substantially oblique direction encountered; simple permutation/flip is insufficient.')
        used.add(current_axis)
        permutation_xyz.append(current_axis)
        flip_xyz.append(D[physical_axis, current_axis] < 0)
    return (tuple(permutation_xyz), tuple(flip_xyz))

def reorient_zyx_to_lps(arr_zyx, direction):
    permutation_xyz, flip_xyz = direction_to_lps_axis_transform(direction)
    arr_xyz = np.transpose(np.asarray(arr_zyx), (2, 1, 0))
    out_xyz = np.transpose(arr_xyz, axes=permutation_xyz)
    for axis, need_flip in enumerate(flip_xyz):
        if need_flip:
            out_xyz = np.flip(out_xyz, axis=axis)
    return np.ascontiguousarray(np.transpose(out_xyz, (2, 1, 0)))

def body_aware_ct_window(ct):
    ct = np.asarray(ct, dtype=np.float32)
    foreground = ct[ct > NON_AIR_THRESHOLD]
    if foreground.size < 100:
        foreground = ct.reshape(-1)
    lo = float(np.percentile(foreground, 10))
    hi = float(np.percentile(foreground, 92))
    if hi <= lo:
        lo = float(np.percentile(ct, 5))
        hi = float(np.percentile(ct, 99))
    return (lo, hi)

def plane_slice(arr, plane, zyx):
    z, y, x = [int(v) for v in zyx]
    if plane == 'Axial':
        return arr[z, :, :]
    if plane == 'Coronal':
        return arr[:, y, :]
    if plane == 'Sagittal':
        return arr[:, :, x]
    raise ValueError(plane)

def plane_origin(plane):
    return 'upper' if plane == 'Axial' else 'lower'

def add_orientation_labels(ax, plane):
    kw = dict(transform=ax.transAxes, fontsize=7.6, fontweight='bold', ha='center', va='center', color='white')
    if plane == 'Axial':
        ax.text(0.02, 0.5, 'R', **kw)
        ax.text(0.98, 0.5, 'L', **kw)
        ax.text(0.5, 0.98, 'A', **kw)
        ax.text(0.5, 0.02, 'P', **kw)
    elif plane == 'Coronal':
        ax.text(0.02, 0.5, 'R', **kw)
        ax.text(0.98, 0.5, 'L', **kw)
        ax.text(0.5, 0.98, 'S', **kw)
        ax.text(0.5, 0.02, 'I', **kw)
    elif plane == 'Sagittal':
        ax.text(0.02, 0.5, 'A', **kw)
        ax.text(0.98, 0.5, 'P', **kw)
        ax.text(0.5, 0.98, 'S', **kw)
        ax.text(0.5, 0.02, 'I', **kw)

def add_contour(ax, mask2d, origin, color, linestyle, linewidth=1.15):
    mask2d = np.asarray(mask2d, dtype=np.float32)
    if mask2d.max() <= 0:
        return
    ax.contour(mask2d, levels=[0.5], origin=origin, colors=[color], linestyles=[linestyle], linewidths=linewidth)

def make_gradcam_alpha(cam2d, gamma=1.35, max_alpha=0.92):
    """
    Display helper only.

    Convert normalized Layer4 Grad-CAM values in [0, 1] to a
    continuous per-pixel alpha map so that low-importance regions
    remain visible but are much more transparent than high-importance
    regions. This improves readability without changing the CAM data.
    """
    cam = np.clip(np.asarray(cam2d, dtype=np.float32), 0.0, 1.0)
    alpha = cam ** gamma * max_alpha
    return np.clip(alpha, 0.0, max_alpha)

def load_screening_case(cohort, patient_id):
    path = SCREEN_ARRAY_DIR / f'{cohort}_TP_{patient_id}_screening_v1.npz'
    require_file(path)
    with np.load(path, allow_pickle=False) as d:
        required = ['ct', 'dose', 'oral', 'gtv', 'patch_direction', 'layer4_mean_cam', 'locked_probability', 'reproduced_probability']
        for k in required:
            if k not in d.files:
                raise KeyError(f'{path.name} missing {k}')
        payload = {'ct': np.asarray(d['ct'], dtype=np.float32), 'dose_normalized': np.asarray(d['dose'], dtype=np.float32), 'oral': np.asarray(d['oral'], dtype=np.float32), 'gtv': np.asarray(d['gtv'], dtype=np.float32), 'patch_direction': np.asarray(d['patch_direction'], dtype=np.float64).reshape(3, 3), 'layer4_mean_cam': np.asarray(d['layer4_mean_cam'], dtype=np.float32), 'locked_probability': float(np.asarray(d['locked_probability']).reshape(-1)[0]), 'reproduced_probability': float(np.asarray(d['reproduced_probability']).reshape(-1)[0]), 'source_array': str(path)}
    return payload

def find_original_npz(patient_id):
    candidates = [ORIGINAL_NPZ_DIR / f'{patient_id}.npz']
    if str(patient_id).isdigit():
        candidates.extend([ORIGINAL_NPZ_DIR / f'{int(patient_id):03d}.npz', ORIGINAL_NPZ_DIR / f'P{int(patient_id):03d}.npz'])
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f'Original NPZ not found for patient {patient_id} in:\n{ORIGINAL_NPZ_DIR}')

def read_scalar_from_npz(data, key):
    if key not in data.files:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if len(arr) == 0:
        return None
    val = float(arr[0])
    if not np.isfinite(val):
        return None
    return val

def recover_physical_dose_gy(patient_id, screening_dose):
    """
    Recover physical dose in Gy from the stored NPZ dose.

    IMPORTANT — exact preprocessing semantics:
    --------------------------------------------
    Original preprocessing did:

        dose_gy = dose_raw * dose_scale_to_gy
        dose_normalized = dose_gy / dose_normalization_gy

    Therefore the NPZ field `dose` is ALREADY normalized.

    Correct inverse transform:

        physical_dose_gy = stored_npz_dose * dose_normalization_gy

    `dose_scale_to_gy` must NOT be multiplied into the stored NPZ dose again.
    It is retained only as provenance/audit metadata describing how the
    ORIGINAL raw RTDOSE was converted to Gy before normalization.
    """
    original_npz = find_original_npz(patient_id)
    with np.load(original_npz, allow_pickle=False) as d:
        normalization_gy = read_scalar_from_npz(d, 'dose_normalization_gy')
        scale_to_gy = read_scalar_from_npz(d, 'dose_scale_to_gy')
        dose_normalized_max_meta = read_scalar_from_npz(d, 'dose_normalized_max')
        stored_original_dose = np.asarray(d['dose'], dtype=np.float32) if 'dose' in d.files else None
    screening_dose = np.asarray(screening_dose, dtype=np.float32)
    if stored_original_dose is None:
        raise RuntimeError(f"Patient {patient_id}: original NPZ does not contain 'dose'.")
    if stored_original_dose.shape != screening_dose.shape:
        raise RuntimeError(f'Patient {patient_id}: screening/original dose shape mismatch.')
    max_abs_diff = float(np.max(np.abs(stored_original_dose - screening_dose)))
    if max_abs_diff > 1e-05:
        raise RuntimeError(f'Patient {patient_id}: screening dose no longer matches original NPZ normalized dose. max |Δ|={max_abs_diff:.8f}')
    if normalization_gy is None or normalization_gy <= 0:
        raise RuntimeError(f'Patient {patient_id}: valid dose_normalization_gy is missing. Cannot recover physical Gy without guessing.')
    physical_dose = (screening_dose * float(normalization_gy)).astype(np.float32)
    dose_min_gy = float(np.min(physical_dose))
    dose_max_gy = float(np.max(physical_dose))
    normalized_max_observed = float(np.max(screening_dose))
    if dose_normalized_max_meta is not None and normalized_max_observed > dose_normalized_max_meta + 0.0001:
        raise RuntimeError(f'Patient {patient_id}: normalized dose max {normalized_max_observed:.6f} exceeds stored dose_normalized_max={dose_normalized_max_meta:.6f}.')
    if not 40.0 <= dose_max_gy <= 95.0:
        raise RuntimeError(f'Patient {patient_id}: recovered max dose = {dose_max_gy:.3f} Gy, outside the expected QC range 40–95 Gy. Stop rather than display an incorrect physical dose scale.')
    return (physical_dose, {'original_npz': str(original_npz), 'scale_source': 'inverse_of_NPZ_normalization: dose_Gy = stored_dose * dose_normalization_gy', 'scale_value_to_gy': float(normalization_gy), 'dose_normalization_gy': float(normalization_gy), 'dose_scale_to_gy': scale_to_gy, 'dose_scale_to_gy_role': 'raw RTDOSE -> Gy before NPZ normalization; not used for inverse transformation of stored NPZ dose', 'dose_normalized_max_metadata': dose_normalized_max_meta, 'dose_normalized_max_observed': normalized_max_observed, 'screening_original_dose_max_abs_difference': max_abs_diff, 'recovered_dose_min_gy': dose_min_gy, 'recovered_dose_max_gy': dose_max_gy})

def rank_sagittal_slice_candidates(cam_lps_zyx, oral_lps_zyx, gtv_lps_zyx):
    """
    Rank sagittal slices for ILLUSTRATIVE display.

    Candidate slices must show both structures and retain substantial
    integrated Layer4 Grad-CAM signal. They are ranked for simultaneous
    oral-cavity/GTV visibility and retained attribution intensity.

    Per sagittal slice x:
      oral_frac[x] = oral area / maximal oral sagittal area
      gtv_frac[x]  = GTV area  / maximal GTV sagittal area
      cam_frac[x]  = integrated CAM / maximal integrated CAM

    Balanced anatomy visibility:
      anatomy_balance[x] = min(oral_frac[x], gtv_frac[x])

    Candidate gate:
      - both oral and GTV must be present;
      - prefer slices retaining >=60% of global integrated CAM;
      - if too few candidates, relax deterministically to
        50%, 40%, 30%, then 20%.

    Visualization score:
      0.55 * anatomy_balance
    + 0.25 * sqrt(oral_frac * gtv_frac)
    + 0.20 * cam_frac

    This scoring is ONLY for choosing an illustrative display slice.
    It is not a statistical endpoint and does not alter Grad-CAM.
    """
    cam = np.asarray(cam_lps_zyx, dtype=np.float32)
    oral = np.asarray(oral_lps_zyx) > 0.5
    gtv = np.asarray(gtv_lps_zyx) > 0.5
    if cam.shape != oral.shape or cam.shape != gtv.shape:
        raise RuntimeError('CAM/oral/GTV shapes do not match.')
    cam_sum = cam.sum(axis=(0, 1)).astype(np.float64)
    oral_area = oral.sum(axis=(0, 1)).astype(np.float64)
    gtv_area = gtv.sum(axis=(0, 1)).astype(np.float64)
    max_cam = float(cam_sum.max())
    max_oral = float(oral_area.max())
    max_gtv = float(gtv_area.max())
    if max_cam <= 0:
        raise RuntimeError('Layer4 CAM is empty.')
    if max_oral <= 0:
        raise RuntimeError('Oral cavity mask is empty.')
    if max_gtv <= 0:
        raise RuntimeError('GTV mask is empty.')
    cam_frac = cam_sum / max_cam
    oral_frac = oral_area / max_oral
    gtv_frac = gtv_area / max_gtv
    anatomy_balance = np.minimum(oral_frac, gtv_frac)
    anatomy_geomean = np.sqrt(oral_frac * gtv_frac)
    both_present = (oral_area > 0) & (gtv_area > 0)
    selected_cam_gate = None
    candidate_mask = None
    for gate in [0.6, 0.5, 0.4, 0.3, 0.2]:
        m = both_present & (cam_frac >= gate)
        if int(m.sum()) >= 3:
            selected_cam_gate = float(gate)
            candidate_mask = m
            break
    if candidate_mask is None:
        candidate_mask = both_present
        selected_cam_gate = 0.0
    if not candidate_mask.any():
        raise RuntimeError('No sagittal slice contains both oral cavity and GTV.')
    visualization_score = 0.55 * anatomy_balance + 0.25 * anatomy_geomean + 0.2 * cam_frac
    rows = []
    for x in np.where(candidate_mask)[0]:
        rows.append({'x_index': int(x), 'oral_area': int(oral_area[x]), 'gtv_area': int(gtv_area[x]), 'oral_fraction_of_max': float(oral_frac[x]), 'gtv_fraction_of_max': float(gtv_frac[x]), 'anatomy_balance': float(anatomy_balance[x]), 'anatomy_geomean': float(anatomy_geomean[x]), 'integrated_cam': float(cam_sum[x]), 'cam_fraction_of_global_max': float(cam_frac[x]), 'visualization_score': float(visualization_score[x]), 'cam_gate_used': float(selected_cam_gate)})
    df = pd.DataFrame(rows).sort_values(['visualization_score', 'anatomy_balance', 'cam_fraction_of_global_max', 'x_index'], ascending=[False, False, False, True]).reset_index(drop=True)
    df['candidate_rank'] = np.arange(1, len(df) + 1)
    x_idx = int(df.iloc[0]['x_index'])
    slice2d = cam[:, :, x_idx]
    z_idx, y_idx = np.unravel_index(int(np.argmax(slice2d)), slice2d.shape)
    selected = df.iloc[0].to_dict()
    return {'zyx': (int(z_idx), int(y_idx), int(x_idx)), 'selected_x_index': int(x_idx), 'candidate_table': df, 'cam_gate_used': float(selected['cam_gate_used']), 'eligible_slice_count': int(len(df)), 'oral_area_selected': int(selected['oral_area']), 'oral_area_max': int(max_oral), 'oral_selected_fraction_of_max': float(selected['oral_fraction_of_max']), 'gtv_area_selected': int(selected['gtv_area']), 'gtv_area_max': int(max_gtv), 'gtv_selected_fraction_of_max': float(selected['gtv_fraction_of_max']), 'anatomy_balance_selected': float(selected['anatomy_balance']), 'integrated_layer4_cam_selected': float(selected['integrated_cam']), 'integrated_layer4_cam_global_max': float(max_cam), 'selected_cam_fraction_of_global_max': float(selected['cam_fraction_of_global_max']), 'visualization_score': float(selected['visualization_score'])}

def choose_sagittal_slice_with_roi_visibility(cam_lps_zyx, oral_lps_zyx, gtv_lps_zyx):
    return rank_sagittal_slice_candidates(cam_lps_zyx, oral_lps_zyx, gtv_lps_zyx)
rank_df = pd.read_csv(require_file(RANKING_CSV))
rank_df['patient_key'] = rank_df['patient_key'].map(normalize_patient_id)
cases = []
audit_rows = []
for cohort, pid, row_header in MAIN_CASES:
    payload = load_screening_case(cohort, pid)
    physical_dose_gy, dose_audit = recover_physical_dose_gy(pid, payload['dose_normalized'])
    direction = payload['patch_direction']
    ct_lps = reorient_zyx_to_lps(payload['ct'], direction)
    dose_gy_lps = reorient_zyx_to_lps(physical_dose_gy, direction)
    oral_lps = reorient_zyx_to_lps(payload['oral'], direction)
    gtv_lps = reorient_zyx_to_lps(payload['gtv'], direction)
    cam_lps = reorient_zyx_to_lps(payload['layer4_mean_cam'], direction)
    selection = choose_sagittal_slice_with_roi_visibility(cam_lps, oral_lps, gtv_lps)
    rank_row = rank_df[(rank_df['cohort'] == cohort) & (rank_df['patient_key'] == pid)].iloc[0]
    if rank_row['category'] != 'TP':
        raise RuntimeError('Figure 4 requires a true-positive case in each cohort; do not label other categories as TP.')
    case = {'cohort': cohort, 'patient_id': pid, 'row_header': row_header, 'ct_lps': ct_lps, 'dose_gy_lps': dose_gy_lps, 'oral_lps': oral_lps, 'gtv_lps': gtv_lps, 'cam_lps': cam_lps, 'locked_probability': float(rank_row['locked_probability']), 'primary_rank': int(rank_row['primary_rank']), 'illustrative_rank': int(rank_row['illustrative_rank']), 'selection': selection, 'dose_audit': dose_audit}
    cases.append(case)
    audit_rows.append({'cohort': cohort, 'patient_id': pid, 'selected_plane': PLANE, 'selected_x_index': selection['selected_x_index'], 'slice_selection_rule': 'among sagittal slices containing both oral cavity and GTV, prefer slices retaining >=60% of the global integrated Layer4 CAM (relaxed deterministically to 50%, 40%, 30%, then 20% if needed), then select rank 1 by the fixed visualization score balancing oral/GTV visibility and retained CAM signal', 'cam_gate_used': selection['cam_gate_used'], 'eligible_slice_count': selection['eligible_slice_count'], 'anatomy_balance_selected': selection['anatomy_balance_selected'], 'visualization_score': selection['visualization_score'], 'oral_area_selected': selection['oral_area_selected'], 'oral_area_max': selection['oral_area_max'], 'oral_selected_fraction_of_max': selection['oral_selected_fraction_of_max'], 'gtv_area_selected': selection['gtv_area_selected'], 'gtv_area_max': selection['gtv_area_max'], 'gtv_selected_fraction_of_max': selection['gtv_selected_fraction_of_max'], 'integrated_layer4_cam_selected': selection['integrated_layer4_cam_selected'], 'integrated_layer4_cam_global_max': selection['integrated_layer4_cam_global_max'], 'selected_cam_fraction_of_global_max': selection['selected_cam_fraction_of_global_max'], 'locked_probability': float(rank_row['locked_probability']), 'primary_rank': int(rank_row['primary_rank']), 'illustrative_rank': int(rank_row['illustrative_rank']), 'dose_scale_source': dose_audit['scale_source'], 'dose_scale_value_to_gy': dose_audit['scale_value_to_gy'], 'dose_normalization_gy': dose_audit['dose_normalization_gy'], 'dose_scale_to_gy': dose_audit['dose_scale_to_gy'], 'recovered_dose_max_gy': dose_audit['recovered_dose_max_gy'], 'original_npz': dose_audit['original_npz']})
slice_audit_df = pd.DataFrame(audit_rows)
slice_audit_path = OUT_DIR / 'Figure4_slice_and_dose_audit.csv'
slice_audit_df.to_csv(slice_audit_path, index=False)

def render_slice_candidate_review(case, top_k=6):
    """
    Show top candidate sagittal slices as Layer4 CAM + anatomy.
    This allows transparent visual review if the automatic rank-1 slice
    is still not the clearest illustrative slice.
    """
    candidate_df = case['selection']['candidate_table'].head(top_k).copy()
    n = len(candidate_df)
    ncols = 3
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.2, 2.9 * nrows))
    axes = np.asarray(axes).reshape(-1)
    ct_vmin, ct_vmax = body_aware_ct_window(case['ct_lps'])
    for i, (_, r) in enumerate(candidate_df.iterrows()):
        ax = axes[i]
        x = int(r['x_index'])
        zyx = (0, 0, x)
        ct2 = plane_slice(case['ct_lps'], PLANE, zyx)
        oral2 = plane_slice(case['oral_lps'], PLANE, zyx)
        gtv2 = plane_slice(case['gtv_lps'], PLANE, zyx)
        cam2 = plane_slice(case['cam_lps'], PLANE, zyx)
        ax.imshow(ct2, cmap='gray', origin='lower', vmin=ct_vmin, vmax=ct_vmax, interpolation='nearest')
        ax.imshow(cam2, cmap=CAM_CMAP, origin='lower', vmin=0, vmax=1, alpha=CAM_ALPHA, interpolation='nearest')
        add_contour(ax, oral2, 'lower', ORAL_COLOR, '-', linewidth=1.0)
        add_contour(ax, gtv2, 'lower', GTV_COLOR, '--', linewidth=1.0)
        add_orientation_labels(ax, PLANE)
        ax.set_title(f"#{int(r['candidate_rank'])}  x={x}\nanatomy={r['anatomy_balance']:.2f}, CAM={r['cam_fraction_of_global_max']:.2f}", fontsize=8.6)
        ax.axis('off')
    for j in range(n, len(axes)):
        axes[j].axis('off')
    fig.suptitle(f"{case['row_header']} — Patient {case['patient_id']} (TP, p={case['locked_probability']:.3f})\nTop sagittal display candidates", fontsize=10.5, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    png = OUT_DIR / f"Patient_{case['patient_id']}_slice_candidates.png"
    pdf = OUT_DIR / f"Patient_{case['patient_id']}_slice_candidates.pdf"
    fig.savefig(png, dpi=400, bbox_inches='tight')
    fig.savefig(pdf, bbox_inches='tight')
    plt.close(fig)
    candidate_df.to_csv(OUT_DIR / f"Patient_{case['patient_id']}_slice_candidates.csv", index=False)
for _case in cases:
    render_slice_candidate_review(_case, top_k=6)
print('=' * 100)
print('=' * 100)
print('\nSLICE / DOSE AUDIT')
print(slice_audit_df.to_string(index=False))
global_dose_max_gy = max((float(np.max(c['dose_gy_lps'])) for c in cases))
shared_dose_vmax_gy = math.ceil(global_dose_max_gy / 5.0) * 5.0
shared_dose_vmin_gy = 0.0
if shared_dose_vmax_gy <= 0:
    raise RuntimeError('Invalid shared dose maximum.')
print(f'\nShared physical dose color scale: {shared_dose_vmin_gy:.0f}–{shared_dose_vmax_gy:.0f} Gy')
plt.rcParams.update({'font.family': 'Arial', 'font.size': 8.7, 'axes.titlesize': 9.4})
fig = plt.figure(figsize=(7.48, 5.35))
gs = fig.add_gridspec(nrows=2, ncols=3, left=0.055, right=0.985, top=0.845, bottom=0.175, wspace=0.035, hspace=0.24)
panel_letters = ['A', 'B', 'C', 'D', 'E', 'F']
column_headers = ['Planning CT', 'Dose distribution', 'Layer4 Grad-CAM']
column_x = [0.205, 0.515, 0.825]
for x, title in zip(column_x, column_headers):
    fig.text(x, 0.925, title, ha='center', va='center', fontsize=9.3, fontweight='bold')
dose_mappable = None
axes_by_row = []
for row_i, case in enumerate(cases):
    zyx = case['selection']['zyx']
    origin = plane_origin(PLANE)
    ct2 = plane_slice(case['ct_lps'], PLANE, zyx)
    dose2 = plane_slice(case['dose_gy_lps'], PLANE, zyx)
    oral2 = plane_slice(case['oral_lps'], PLANE, zyx)
    gtv2 = plane_slice(case['gtv_lps'], PLANE, zyx)
    cam2 = plane_slice(case['cam_lps'], PLANE, zyx)
    ct_vmin, ct_vmax = body_aware_ct_window(case['ct_lps'])
    row_axes = []
    for col_i in range(3):
        ax = fig.add_subplot(gs[row_i, col_i])
        row_axes.append(ax)
        ax.imshow(ct2, cmap='gray', origin=origin, vmin=ct_vmin, vmax=ct_vmax, interpolation='nearest')
        if col_i == 1:
            dose_mappable = ax.imshow(dose2, cmap=DOSE_CMAP, origin=origin, interpolation='nearest', alpha=DOSE_ALPHA, vmin=shared_dose_vmin_gy, vmax=shared_dose_vmax_gy)
        elif col_i == 2:
            ax.imshow(cam2, cmap=CAM_CMAP, origin=origin, interpolation='nearest', alpha=CAM_ALPHA, vmin=0, vmax=1)
        add_contour(ax, oral2, origin, ORAL_COLOR, '-', linewidth=1.15)
        add_contour(ax, gtv2, origin, GTV_COLOR, '--', linewidth=1.15)
        add_orientation_labels(ax, PLANE)
        ax.text(0.018, 0.975, panel_letters[row_i * 3 + col_i], transform=ax.transAxes, fontsize=11.0, fontweight='bold', va='top', ha='left', color='white', bbox=dict(boxstyle='round,pad=0.12', facecolor='black', edgecolor='none', alpha=0.55))
        ax.axis('off')
    axes_by_row.append(row_axes)
top_row_y = axes_by_row[0][0].get_position().y1 + 0.018
bottom_row_y = axes_by_row[1][0].get_position().y1 + 0.018
for case, y in zip(cases, [top_row_y, bottom_row_y]):
    fig.text(0.055, y, f"{case['row_header']} — Representative TP case (p = {case['locked_probability']:.3f})", ha='left', va='bottom', fontsize=9.0, fontweight='bold')
legend_handles = [Line2D([0], [0], color=ORAL_COLOR, linewidth=1.8, linestyle='-', label='Oral cavity'), Line2D([0], [0], color=GTV_COLOR, linewidth=1.8, linestyle='--', label='GTV')]
fig.legend(handles=legend_handles, loc='lower left', bbox_to_anchor=(0.055, 0.045), ncol=2, frameon=False, fontsize=7.9, handlelength=2.4, columnspacing=1.7)
dose_cax = fig.add_axes([0.38, 0.07, 0.2, 0.016])
dose_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=shared_dose_vmin_gy, vmax=shared_dose_vmax_gy), cmap=DOSE_CMAP)
dose_sm.set_array([])
dose_cb = fig.colorbar(dose_sm, cax=dose_cax, orientation='horizontal')
dose_cb.set_label('Dose (Gy)', fontsize=7.7, labelpad=1.5)
dose_cb.set_ticks(np.linspace(shared_dose_vmin_gy, shared_dose_vmax_gy, 5))
dose_cb.ax.tick_params(labelsize=7.0, length=2, pad=1)
cam_cax = fig.add_axes([0.705, 0.07, 0.2, 0.016])
cam_sm = plt.cm.ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=CAM_CMAP)
cam_sm.set_array([])
cam_cb = fig.colorbar(cam_sm, cax=cam_cax, orientation='horizontal')
cam_cb.set_ticks([0, 1])
cam_cb.set_ticklabels(['Low', 'High'])
cam_cb.set_label('Relative Grad-CAM importance', fontsize=7.7, labelpad=1.5)
cam_cb.ax.tick_params(labelsize=7.0, length=2, pad=1)
png_path = OUT_DIR / 'Figure4_RnO.png'
tiff_path = OUT_DIR / 'Figure4_RnO.tiff'
pdf_path = OUT_DIR / 'Figure4_RnO.pdf'
fig.savefig(png_path, dpi=600, bbox_inches='tight')
fig.savefig(tiff_path, dpi=600, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
plt.show()
plt.close(fig)
legend_text = 'Figure 4. Spatial interpretation of the M3 model in representative true-positive cases. Panels A–C show a representative true-positive case from the development cohort, and panels D–F show a representative true-positive case from the independent external validation cohort. For each case, the sagittal display slice was selected using a fixed visualization ranking that balanced simultaneous oral-cavity/GTV visibility with retention of integrated Layer4 Grad-CAM signal. Panels A and D show planning CT with the oral cavity (green solid contour) and GTV (cyan dashed contour). Panels B and E show the physical dose distribution in Gy using a shared color scale. Panels C and F show Layer4 Grad-CAM overlaid on the planning CT. Warm Grad-CAM colors indicate relatively stronger positive contributions to the severe-mucositis logit within each case. Grad-CAM represents coarse spatial attribution rather than voxel-level localization. CT was used only as an anatomical reference and was not an input to M3.'
with open(OUT_DIR / 'Figure4_RnO_legend.txt', 'w', encoding='utf-8') as f:
    f.write(legend_text)
audit = {'version': 'Figure4_RnO', 'main_cases': [f'Development TP case {DEVELOPMENT_CASE_ID}', f'External TP case {EXTERNAL_CASE_ID}'], 'case_selection_source': 'configured Grad-CAM case selection', 'display_plane': 'Sagittal', 'slice_selection': {'purpose': 'illustrative display selection only', 'plane': 'sagittal', 'candidate_gate': 'both oral cavity and GTV present; prefer slices retaining >=60% of global integrated Layer4 CAM, relaxed deterministically to 50%, 40%, 30%, then 20% if fewer than 3 candidates', 'visualization_score': '0.55*min(oral_fraction_of_max, gtv_fraction_of_max) + 0.25*sqrt(oral_fraction_of_max*gtv_fraction_of_max) + 0.20*cam_fraction_of_global_max', 'note': 'The rule affects visualization only; CAM is not constrained to lie inside oral cavity or GTV.', 'candidate_review_output': True}, 'dose_display': {'unit': 'Gy', 'source': 'stored NPZ dose is dose_Gy / dose_normalization_gy; physical Gy is recovered exactly as stored_dose * dose_normalization_gy. dose_scale_to_gy is provenance for raw RTDOSE conversion only.', 'shared_vmin_gy': float(shared_dose_vmin_gy), 'shared_vmax_gy': float(shared_dose_vmax_gy), 'shared_scale_for_both_cases': True}, 'primary_gradcam_layer': 'layer4', 'gradcam_display': {'colormap': CAM_CMAP, 'purpose': 'so hue has a direct one-dimensional correspondence to normalized CAM value'}, 'main_columns': ['Planning CT', 'Dose distribution', 'Layer4 Grad-CAM'], 'consensus_in_main_figure': False, 'layer3_in_main_figure': False, 'cam_sd_in_main_figure': False, 'no_retraining': True, 'no_reinference': True, 'no_recalibration': True, 'no_threshold_tuning': True, 'status': 'PASS'}
with open(OUT_DIR / 'Figure4_RnO_audit.json', 'w', encoding='utf-8') as f:
    json.dump(audit, f, ensure_ascii=False, indent=2)
print('\n' + '=' * 100)
print(f'Output folder:\n{OUT_DIR}')
print('\nGenerated files:')
print('  1) Figure4_RnO.pdf')
print('  2) Figure4_RnO.tiff')
print('  3) Figure4_RnO.png')
print('  4) Figure4_slice_and_dose_audit.csv')
print('  5) Figure4_RnO_legend.txt')
print('  6) Figure4_RnO_audit.json')
print('=' * 100)
register('figure4_dir', str(OUT_DIR))
