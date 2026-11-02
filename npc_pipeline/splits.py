"""Generate or import the locked 4 x 5 development splits using final cohort inputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from .common import config, load_cohort, require, new_output, register, sha256, write_json, identity_digest, integer_values

PARTS = ('outer_train', 'outer_validation', 'inner_train', 'inner_validation')
OUTER_SEEDS = (20260701, 20260702, 20260703, 20260704)

def generated_frames(included):
    dev = included.loc[included.model_center.isin(['B', 'C'])].sort_values('patient_id').reset_index(drop=True)
    frames, configs = {}, {}
    for repeat, seed in enumerate(OUTER_SEEDS, 1):
        outer = StratifiedKFold(5, shuffle=True, random_state=seed)
        for fold, (ot, ov) in enumerate(outer.split(dev, dev.stratum), 1):
            order = ['model_center', 'severe_mucositis', 'patient_id']
            train = dev.iloc[ot].sort_values(order).reset_index(drop=True)
            valid = dev.iloc[ov].sort_values(order).reset_index(drop=True)
            inner_seed, train_seed = 50260700 + repeat * 10 + fold, 30260700 + repeat * 10 + fold
            inner = StratifiedShuffleSplit(1, test_size=.2, random_state=inner_seed)
            it, iv = next(inner.split(train, train.stratum))
            frames[repeat, fold] = dict(outer_train=train, outer_validation=valid,
                inner_train=train.iloc[it].sort_values(order).reset_index(drop=True),
                inner_validation=train.iloc[iv].sort_values(order).reset_index(drop=True))
            configs[repeat, fold] = {'repeat_number': repeat, 'fold_number': fold,
                'outer_split_seed': seed, 'inner_split_seed': inner_seed, 'training_seed': train_seed,
                'external_n': 180, 'external_center': 'A',
                **{k + '_n': len(v) for k, v in frames[repeat, fold].items()}}
    return frames, configs

def read_parts(root, included, compare_generated=True):
    dev = included.loc[included.model_center.isin(['B', 'C'])].set_index('patient_id')
    expected, configs = generated_frames(included)
    frames, files = {}, {}
    for key in expected:
        folder = Path(root) / f'repeat_{key[0]:02d}' / f'fold_{key[1]:02d}'
        if (folder / 'locked_split_inputs').is_dir():
            folder = folder / 'locked_split_inputs'
        cfg_path = folder / 'fold_config.json'
        actual_cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
        for k in ('repeat_number', 'fold_number', 'outer_split_seed', 'inner_split_seed', 'training_seed'):
            require(actual_cfg[k] == configs[key][k], f'{folder}: wrong {k}')
        parts = {}
        for role in PARTS:
            path = folder / (role + '.csv')
            z = pd.read_csv(path)
            ids = integer_values(z.patient_id, str(path))
            require(not ids.duplicated().any() and set(ids).issubset(dev.index), f'{path}: duplicate/non-development patient')
            ref = dev.loc[ids].reset_index()
            require(np.array_equal(z.severe_mucositis.to_numpy(), ref.severe_mucositis.to_numpy()), f'{path}: outcome mismatch')
            require(np.array_equal(z.model_center.to_numpy(), ref.model_center.to_numpy()), f'{path}: center mismatch')
            if compare_generated:
                require(set(ids) == set(expected[key][role].patient_id), f'{path}: does not match the fixed split algorithm/seeds')
            parts[role] = ref
            files[f'repeat_{key[0]:02d}/fold_{key[1]:02d}/{role}.csv'] = path
        sets = {k: set(v.patient_id) for k, v in parts.items()}
        require(not sets['outer_train'] & sets['outer_validation'] and sets['outer_train'] | sets['outer_validation'] == set(dev.index), 'Outer split leakage/incomplete coverage')
        require(not sets['inner_train'] & sets['inner_validation'] and sets['inner_train'] | sets['inner_validation'] == sets['outer_train'], 'Inner split leakage/incomplete coverage')
        frames[key] = parts
        files[f'repeat_{key[0]:02d}/fold_{key[1]:02d}/fold_config.json'] = cfg_path
    for r in range(1, 5):
        ids = [p for (rr, _), parts in frames.items() if rr == r for p in parts['outer_validation'].patient_id]
        require(len(ids) == len(set(ids)) == 309, 'Each repeat must have exactly one OOF assignment per development patient')
    return frames, configs, files

def run(import_root=None):
    c = config()
    _, d = load_cohort(c)
    require('npz_path' in d, 'Set npz_dir before generating splits')
    frames, configs = generated_frames(d)
    if import_root:
        frames, configs, sources = read_parts(import_root, d)
    else:
        sources = {}
    out = new_output('Splits_final489')
    cols = ['patient_id', 'model_center', 'severe_mucositis', 'cohort', 'stratum', 'npz_path']
    for (r, f), parts in frames.items():
        folder = out / f'repeat_{r:02d}' / f'fold_{f:02d}'
        folder.mkdir(parents=True)
        for role, df in parts.items():
            df[cols].to_csv(folder / f'{role}.csv', index=False)
        write_json(folder / 'fold_config.json', configs[r, f])
    hashes = {str(p.relative_to(out)): sha256(p) for p in out.rglob('*') if p.is_file()}
    write_json(out / 'SPLITS_LOCKED.json', {'identity_digest': identity_digest(d), 'master_sha256': sha256(c['master']),
        'cohort': {'assessed': 540, 'excluded': 51, 'included': 489, 'development': 309, 'external': 180},
        'file_hashes': hashes, 'imported_sources': {str(p): sha256(p) for p in sources.values()}})
    register('split_dir', str(out))
    print('Final 489 split lock:', out)
    return out

def load_locked(c=None):
    c = c or config()
    _, d = load_cohort(c)
    root = Path(c['split_dir'])
    lock = json.loads((root / 'SPLITS_LOCKED.json').read_text(encoding='utf-8'))
    require(lock['identity_digest'] == identity_digest(d), 'Master cohort identity/outcome changed after split lock')
    for rel, h in lock['file_hashes'].items():
        p = (root / rel).resolve()
        require(p.is_relative_to(root.resolve()) and p.is_file() and sha256(p) == h, 'Split file missing/changed: ' + rel)
    return read_parts(root, d)
