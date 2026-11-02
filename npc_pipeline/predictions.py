"""Align frozen predictions to final master; expose one consistent analysis layout."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from .common import config, load_cohort, checked_predictions, require, new_output, register, sha256, write_json
from .splits import load_locked

ORIGINAL = {'M1': 'M1_Dose_Only', 'M2_Abl': 'M2_Abl', 'M2': 'M2_Oral_MaskedDose',
    'M3_Abl': 'M3_Abl', 'M3': 'M3_Oral_GTV_MaskedDose', 'M4': 'M4_CT_Augmented_M3'}
REVISION = {'MasksOnly': 'R1A_MasksOnly', 'DoseGTV': 'R1B_DoseGTV'}


def check_clinical_ensemble(oof, external, dev_mean, ext_mean, refs, frames):
    """Validate every held-out assignment and every external ensemble member."""
    require(len(oof) == 1236 and not oof.duplicated(['repeat', 'patient_id']).any(),
            'Clinical OOF table must contain four predictions per development patient')
    require(len(external) == 3600 and not external.duplicated(['repeat', 'fold', 'patient_id']).any(),
            'Clinical external table must contain 20 predictions per external patient')
    for z in (oof, external):
        require(set(zip(z['repeat'], z['fold'])) == set(frames), 'Clinical fold set mismatch')
    for (r, f), parts in frames.items():
        checked_predictions(oof[(oof['repeat'] == r) & (oof['fold'] == f)],
                            parts['outer_validation'].sort_values('patient_id'))
        e = external[(external['repeat'] == r) & (external['fold'] == f)].copy()
        # The external long table stores predictions; outcomes come from the final master.
        if 'y_true' not in e:
            e['y_true'] = e.patient_id.map(refs['External'].set_index('patient_id').severe_mucositis)
        checked_predictions(e, refs['External'])
    for name, long, mean in [('Development', oof, dev_mean), ('External', external, ext_mean)]:
        expected = long.groupby('patient_id').probability.mean().reindex(refs[name].patient_id)
        require(np.allclose(expected, mean.probability, atol=1e-10, rtol=1e-8),
                f'Clinical {name} mean does not match per-fold predictions')


def export():
    c = config()
    _, included = load_cohort(c)
    frames, _, _ = load_locked(c)
    refs = {'Development': included[included.model_center.isin(['B', 'C'])].copy(),
            'External': included[included.model_center.eq('A')].copy()}
    sources = c.get('frozen_sources', {})
    out = new_output('Frozen_predictions_final489')
    (out / 'development').mkdir()
    (out / 'external').mkdir()
    source_hashes, all_frames, oof_long = {}, {}, {}
    def read(path):
        path = Path(path)
        source_hashes[str(path)] = sha256(path)
        return pd.read_csv(path)
    for display, model in {**ORIGINAL, **REVISION}.items():
        run = c.get('cnn_runs', {}).get(model) or sources.get('revision_runs', {}).get(display)
        if run:
            run = Path(run)
            marker = json.loads((run / 'RUN_COMPLETED.json').read_text(encoding='utf-8'))
            require(marker.get('status') == 'PASS' and marker.get('n_models') == 20, 'CNN run not complete: ' + str(run))
            pair = [read(run / 'aggregate/Development_patient_predictions_309.csv'),
                    read(run / 'aggregate/External_patient_predictions_180.csv')]
            for frame, n in zip(pair, [4, 20]):
                require(frame.prediction_count.eq(n).all(), 'Incorrect ensemble size')
            long = read(run / 'aggregate/Development_OOF_long_1236.csv')
        elif display in ORIGINAL and sources.get('development_predictions') and sources.get('external_predictions'):
            dev_dir, ext_dir = Path(sources['development_predictions']), Path(sources['external_predictions'])
            pair = [read(dev_dir / f'{model}_patient_averaged_oof_309.csv'),
                    read(ext_dir / f'{model}_CenterA_180_20model_ensemble_predictions.csv')]
            require(pair[0].oof_prediction_count.eq(4).all() and pair[1].n_checkpoints.eq(20).all(), 'Incorrect frozen ensemble size')
            pair[0]['probability'] = pair[0]['oof_probability_mean']
            long = read(dev_dir / f'{model}_oof_long_1236.csv')
        else:
            continue
        require(len(long) == 1236 and not long.duplicated(['repeat', 'patient_id']).any(), 'Incomplete OOF long table')
        require(set(zip(long['repeat'], long['fold'])) == set(frames), 'Unexpected OOF repeat/fold')
        for (r, f), parts in frames.items():
            z = long[(long['repeat'] == r) & (long['fold'] == f)]
            checked_predictions(z, parts['outer_validation'].sort_values('patient_id'))
        for cohort, frame in zip(refs, pair):
            aligned = checked_predictions(frame, refs[cohort])
            all_frames[cohort, display] = aligned
        means = long.groupby('patient_id').probability.mean().reindex(refs['Development'].patient_id)
        require(np.allclose(means, all_frames['Development', display].probability, atol=1e-10, rtol=1e-8), 'OOF mean differs from frozen predictions')
        oof_long[display] = long
    clinical = c.get('clinical_run') or sources.get('clinical_run')
    if clinical:
        marker = json.loads((Path(clinical) / 'RUN_COMPLETED.json').read_text(encoding='utf-8'))
        require(marker.get('status') == 'COMPLETED' and marker.get('models_per_comparator') == 20, 'Clinical run not complete')
        for cohort in refs:
            table = read(Path(clinical) / f'{cohort}_patient_predictions.csv')
            for label, col in [('C1', 'C1_Clinical_Chemo_DVH'), ('C2', 'C2_Clinical_Chemo_DVH_GTV')]:
                all_frames[cohort, label] = checked_predictions(table, refs[cohort], col)
        clinical_oof = read(Path(clinical) / 'comparator_oof_long.csv')
        clinical_ext = read(Path(clinical) / 'comparator_external_20model_long.csv')
        for label, col in [('C1', 'C1_Clinical_Chemo_DVH'), ('C2', 'C2_Clinical_Chemo_DVH_GTV')]:
            check_clinical_ensemble(clinical_oof[clinical_oof.model.eq(col)],
                clinical_ext[clinical_ext.model.eq(col)], all_frames['Development', label],
                all_frames['External', label], refs, frames)
    require(all_frames, 'No complete predictions configured')
    models = sorted({key[1] for key in all_frames})
    external_rows = []
    for cohort, ref in refs.items():
        wide = ref[['patient_id', 'model_center', 'severe_mucositis']].rename(columns={'severe_mucositis': 'y_true'}).reset_index(drop=True)
        for display in models:
            frame = all_frames[cohort, display]
            wide[display] = frame.probability.to_numpy()
            frame[['patient_id', 'model_center', 'y_true', 'probability']].to_csv(out / f'{cohort}_{display}.csv', index=False)
            if display in ORIGINAL:
                name = ORIGINAL[display]
                if cohort == 'Development':
                    z = frame[['patient_id', 'model_center', 'y_true', 'probability']].rename(columns={'probability': 'oof_probability_mean'})
                    z['oof_prediction_count'] = 4
                    z.to_csv(out / 'development' / f'{name}_patient_averaged_oof_309.csv', index=False)
                    oof_long[display].to_csv(out / 'development' / f'{name}_oof_long_1236.csv', index=False)
                else:
                    z = frame[['patient_id', 'model_center', 'y_true', 'probability']].copy()
                    z['n_checkpoints'] = 20
                    z.to_csv(out / 'external' / f'{name}_CenterA_180_20model_ensemble_predictions.csv', index=False)
                    z['display_name'] = display.replace('_Abl', '-Abl')
                    external_rows.append(z)
        wide.to_csv(out / f'{cohort}_aligned_predictions.csv', index=False)
    if external_rows:
        pd.concat(external_rows).to_csv(out / 'external/external_patient_20model_ensemble_all_models.csv', index=False)
    write_json(out / 'prediction_audit.json', {'status': 'PASS', 'models': models, 'source_hashes': source_hashes,
        'export_hashes': {str(p.relative_to(out)): sha256(p) for p in out.rglob('*.csv')},
        'master_sha256': sha256(c['master']), 'split_lock_sha256': sha256(Path(c['split_dir']) / 'SPLITS_LOCKED.json')})
    register('prediction_dir', str(out))
    print('Aligned frozen predictions:', out)
    return out

def load_aligned(require_all=False):
    c = config()
    root = Path(c['prediction_dir'])
    audit = json.loads((root / 'prediction_audit.json').read_text(encoding='utf-8'))
    for rel, h in audit['export_hashes'].items():
        path = (root / rel).resolve()
        require(path.is_relative_to(root.resolve()) and path.is_file() and sha256(path) == h, 'Frozen prediction export changed: ' + rel)
    require(audit['master_sha256'] == sha256(c['master']), 'Master changed since prediction export; export again')
    require(audit['split_lock_sha256'] == sha256(Path(c['split_dir']) / 'SPLITS_LOCKED.json'), 'Split lock changed since prediction export')
    _, d = load_cohort(c)
    tables = {}
    for cohort, centers in [('Development', ['B', 'C']), ('External', ['A'])]:
        z = pd.read_csv(root / f'{cohort}_aligned_predictions.csv')
        ref = d[d.model_center.isin(centers)]
        for model in audit['models']:
            checked_predictions(z, ref, model)
        if require_all:
            require(set(ORIGINAL) | set(REVISION) | {'C1', 'C2'} <= set(z), 'All 10 models are needed for the complete revision comparison')
        tables[cohort] = z
    return tables
