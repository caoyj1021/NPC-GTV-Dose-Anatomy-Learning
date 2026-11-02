"""Final-cohort spatial preparation. Dose conversion requires explicit per-case QC."""
from pathlib import Path
import json
import uuid
import numpy as np
import pandas as pd
import SimpleITK as sitk
from .common import config, load_cohort, require, new_output, register, sha256, write_json, integer_values
from . import spatial_core as core

def resolve_image(value, patient_id):
    path = Path(value)
    if path.is_file():
        return path
    if path.is_dir():
        matches = [p for p in (path / f'{patient_id}.nii.gz', path / f'{patient_id}.nii') if p.is_file()]
        require(len(matches) == 1, f'{path}: expected exactly one {patient_id}.nii[.gz]; do not guess among files')
        return matches[0]
    raise FileNotFoundError(path)

def prepare_case(row, dose_scale):
    require(np.isfinite(dose_scale) and 0 < dose_scale <= 10, 'Explicit raw dose-to-Gy factor required')
    images = {k: core.read_3d_image(resolve_image(row[col], row['patient_id'])) for k, col in {
        'ct': 'ct_path', 'dose': 'dose_path', 'oral': 'oral_cavity_mask_path', 'gtv': 'gtv_mask_path'}.items()}
    for k, im in images.items():
        require(core.same_geometry(images['ct'], im), f'{k}: source geometry differs from CT; resolve upstream alignment')
        require(np.isfinite(sitk.GetArrayViewFromImage(im)).all(), f'{k}: non-finite source pixels')
    masks = {k: sitk.Cast(images[k] > 0, sitk.sitkUInt8) for k in ('oral', 'gtv')}
    center, center_index = core.mask_bbox_center_physical(masks['oral'])
    reference = core.create_patch_reference(images['ct'], center)
    retention = {k: core.patch_retained_fraction(v, reference) for k, v in masks.items()}
    require(retention['oral'] >= .999, 'Oral mask retention is below 99.9%')
    require(retention['gtv'] >= .95, 'GTV retention below 95%; do not accept without revisiting crop')
    ct = core.resample_to_reference(images['ct'], reference, sitk.sitkLinear, -1000., sitk.sitkFloat32)
    dose = core.resample_to_reference(images['dose'], reference, sitk.sitkLinear, 0., sitk.sitkFloat32)
    oral, gtv = [core.resample_to_reference(masks[k], reference, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8) for k in ('oral', 'gtv')]
    arrays = core.prepare_arrays(ct, dose, oral, gtv, dose_scale)
    metadata = {'patient_id': int(row['patient_id']), 'model_center': row['model_center'],
        'cohort': row['cohort'], 'severe_mucositis': int(row['severe_mucositis']),
        'label': int(row['severe_mucositis']), 'patch_direction': reference.GetDirection(),
        'patch_origin_xyz': reference.GetOrigin(), 'oral_center_physical_xyz': center,
        'target_spacing_xyz': core.TARGET_SPACING_XYZ, 'patch_size_xyz': core.PATCH_SIZE_XYZ,
        'dose_normalization_gy': 70., 'dose_normalized_max': 1.2, 'dose_scale_to_gy': float(dose_scale)}
    return arrays, metadata, retention

def save_npz(path, arrays, metadata):
    path = Path(path)
    require(not path.exists(), f'Refusing overwrite: {path}')
    # A temporary FILE in the destination directory, never the directory itself.
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with tmp.open('xb') as stream:
            np.savez_compressed(stream, **{k: arrays[k] for k in ('ct', 'dose', 'oral', 'gtv')}, **metadata)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

def run():
    c = config()
    _, included = load_cohort(c)
    scale_path = Path(c['dose_scale_csv'])
    scale = pd.read_csv(scale_path)
    scale['patient_id'] = integer_values(scale.patient_id, 'dose scale patient_id')
    require(not scale.patient_id.duplicated().any(), 'Duplicate dose QC IDs')
    require(set(included.patient_id).issubset(scale.patient_id), 'Dose-scale QC is missing final included IDs')
    scale = scale.set_index('patient_id')
    require('dose_scale_to_gy' in scale, 'Dose QC requires dose_scale_to_gy; do not infer units from values')
    approved = {}
    if c.get('coverage_approvals_csv'):
        a = pd.read_csv(c['coverage_approvals_csv'])
        require(not a.patient_id.duplicated().any(), 'Duplicate coverage-approval IDs')
        approved = a.set_index('patient_id').to_dict('index')
    out = new_output('Preprocessing_final489')
    (out / 'npz').mkdir()
    records = []
    for n, row in enumerate(included.to_dict('records'), 1):
        pid = row['patient_id']
        record = {'patient_id': pid, 'model_center': row['model_center'], 'status': 'ERROR'}
        try:
            arrays, meta, retention = prepare_case(row, float(scale.loc[pid, 'dose_scale_to_gy']))
            save_npz(out / 'npz' / f'{pid}.npz', arrays, meta)
            record.update(oral_retained_fraction=retention['oral'], gtv_retained_fraction=retention['gtv'])
            warning = retention['gtv'] < .99
            approval = approved.get(pid, {})
            accepted = (approval.get('approved') == 1 and
                        abs(float(approval.get('gtv_retained_fraction', -1)) - retention['gtv']) < 1e-5)
            record['status'] = 'REVIEW_REQUIRED' if warning and not accepted else 'PASS'
            record['manual_approval_used'] = bool(warning and accepted)
            record['npz_sha256'] = sha256(out / 'npz' / f'{pid}.npz')
        except Exception as exc:
            record['error'] = str(exc)
        records.append(record)
        if n % 25 == 0 or n == len(included):
            print(f'Processed {n}/{len(included)}', flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'preprocessing_QC.csv', index=False, encoding='utf-8-sig')
    frame.to_excel(out / 'preprocessing_QC.xlsx', index=False)
    passed = frame.status.eq('PASS').all()
    write_json(out / 'status.json', {'status': 'PASS' if passed else 'REVIEW_REQUIRED', 'n': len(frame),
        'master_sha256': sha256(c['master']), 'dose_scale_sha256': sha256(scale_path),
        'coverage_approval_sha256': sha256(c['coverage_approvals_csv']) if c.get('coverage_approvals_csv') else None})
    require(passed, f'Preprocessing requires review; inspect {out}. Input config was not advanced.')
    register('npz_dir', str(out / 'npz'))
    return out

def coverage_qc():
    """Geometric source-mask retention; warning cases remain visible for manual review."""
    c = config()
    _, included = load_cohort(c)
    out = new_output('Patch_coverage_final489')
    rows = []
    for row in included.to_dict('records'):
        record = {'patient_id': row['patient_id'], 'model_center': row['model_center'], 'status': 'ERROR'}
        try:
            ct = core.read_3d_image(resolve_image(row['ct_path'], row['patient_id']))
            oral = core.read_3d_image(resolve_image(row['oral_cavity_mask_path'], row['patient_id']))
            gtv = core.read_3d_image(resolve_image(row['gtv_mask_path'], row['patient_id']))
            require(core.same_geometry(ct, oral) and core.same_geometry(ct, gtv), 'Source-mask geometry mismatch')
            center, _ = core.mask_bbox_center_physical(oral)
            reference = core.create_patch_reference(ct, center)
            o, g = core.patch_retained_fraction(oral, reference), core.patch_retained_fraction(gtv, reference)
            require(np.isfinite(o) and np.isfinite(g), 'Invalid retention fractions')
            record.update(oral_retained_fraction=o, gtv_retained_fraction=g,
                status='PASS' if o >= .999 and g >= .99 else 'REVIEW_REQUIRED')
        except Exception as exc:
            record['error'] = str(exc)
        rows.append(record)
        if len(rows) % 25 == 0:
            print(f'Coverage: {len(rows)}/489', flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'patch_coverage.csv', index=False, encoding='utf-8-sig')
    frame.to_excel(out / 'patch_coverage.xlsx', index=False)
    write_json(out / 'summary.json', {'n': len(frame), 'status_counts': frame.status.value_counts().to_dict(),
        'master_sha256': sha256(c['master']), 'note': 'Warnings require clinical review, not automatic patient exclusion.'})
    register('coverage_qc_dir', str(out))
    return out

def audit_existing_npz():
    c = config()
    _, included = load_cohort(c)
    out = new_output('NPZ_audit_final489')
    rows = []
    for row in included.itertuples():
        path = Path(row.npz_path)
        with np.load(path, allow_pickle=False) as z:
            internal_id = np.asarray(z['patient_id'])
            require(internal_id.size == 1 and np.isfinite(internal_id).all() and float(internal_id.item()) == row.patient_id, 'NPZ identity mismatch')
            for channel in ('ct', 'dose', 'oral', 'gtv'):
                a = z[channel]
                require(a.shape == (64, 112, 80) and np.isfinite(a).all(), f'{path}: {channel} invalid')
                if channel in ('oral', 'gtv'):
                    require(np.isin(a, [0, 1]).all() and a.any(), f'{path}: mask invalid')
            require(z['ct'].min() >= -1.0001 and z['ct'].max() <= 1.0001, 'CT normalization mismatch')
            require(z['dose'].min() >= -1e-6 and z['dose'].max() <= 1.2001, 'Dose normalization mismatch')
        rows.append({'patient_id': row.patient_id, 'sha256': sha256(path)})
    pd.DataFrame(rows).to_csv(out / 'npz_hashes.csv', index=False)
    # Extra files may exist, but selection is exclusively by the final master IDs.
    extras = sorted(p.name for p in Path(c['npz_dir']).glob('*.npz') if p.stem not in set(included.patient_id.astype(str)))
    write_json(out / 'status.json', {'status': 'PASS', 'selected_n': 489, 'unselected_npz_files': extras,
        'label_source': 'final master', 'npz_label_and_cohort_read': False, 'scope': 'Array integrity, not source-mask retention QC'})
    register('npz_audit_dir', str(out))
    return out
