"""Final-cohort configuration, identity checks, and non-overwriting outputs."""
from pathlib import Path
from datetime import datetime
import hashlib
import json
import os
import uuid
import numpy as np
import pandas as pd

COUNTS = {('A', 0): 88, ('A', 1): 92, ('B', 0): 42, ('B', 1): 39,
          ('C', 0): 116, ('C', 1): 112}
ROLES = {'A': 'External', 'B': 'Development', 'C': 'Development'}
CENTERS = {'A': 'Jiaxing', 'B': 'Yantai', 'C': 'Ningbo'}
EXCLUSIONS = {'M1 metastatic disease': 28, 'missing CT RTPLAN': 20, 'no_mask_gtv': 3}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def config_file():
    if os.environ.get('NPC_CONFIG'):
        return Path(os.environ['NPC_CONFIG']).expanduser().resolve()
    for parent in [Path.cwd(), *Path.cwd().parents]:
        p = parent / 'config.local.json'
        if p.is_file():
            return p
    raise FileNotFoundError('Copy config.example.json to config.local.json and fill the input paths, or set NPC_CONFIG.')

def config():
    c = json.loads(config_file().read_text(encoding='utf-8-sig'))
    for k in ['master', 'result_root']:
        require(k in c and Path(c[k]).is_absolute(), f'{k} must be an explicit absolute path')
    return c

def register(key, value):
    c = config()
    c[key] = value
    write_json(config_file(), c)

def new_output(stage):
    root = Path(config()['result_root'])
    root.mkdir(parents=True, exist_ok=True)
    out = root / f'{stage}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    out.mkdir(exist_ok=False)
    return out

def integer_values(series, field):
    x = pd.to_numeric(series, errors='raise')
    require(x.notna().all() and np.isfinite(x).all() and (x == np.floor(x)).all(), f'{field}: non-integer/missing values')
    return x.astype('int64')

def validate_cohort(raw):
    raw = raw.copy()
    required = {'patient_id', 'model_center', 'severe_mucositis', 'exclude_reason'}
    require(required.issubset(raw.columns), f'Missing columns: {required - set(raw.columns)}')
    require(len(raw) == 540, f'Expected 540 assessed records, found {len(raw)}')
    raw['patient_id'] = integer_values(raw.patient_id, 'patient_id')
    require(not raw.patient_id.duplicated().any(), 'Duplicate patient_id')
    reason = raw.exclude_reason.fillna('').astype(str).str.strip()
    require(reason[reason.ne('')].value_counts().to_dict() == EXCLUSIONS, 'Expected the final 51 exclusions (28/20/3)')
    included = raw.loc[reason.eq('')].copy()
    included['model_center'] = included.model_center.astype(str).str.strip().str.upper()
    included['severe_mucositis'] = integer_values(included.severe_mucositis, 'severe_mucositis')
    require(included.groupby(['model_center', 'severe_mucositis']).size().to_dict() == COUNTS,
            'Final cohort mismatch: 489 included; A 88/92, B 42/39, C 116/112')
    wanted = included.model_center.map(ROLES)
    if 'cohort' in included:
        require(included.cohort.astype(str).str.strip().eq(wanted).all(),
                'cohort conflicts with the final B+C development / A external allocation; correct the source explicitly')
    included['cohort'] = wanted
    included['stratum'] = included.model_center + '_label' + included.severe_mucositis.astype(str)
    included = included.sort_values('patient_id').reset_index(drop=True)
    return raw, included

def load_cohort(c=None):
    c = c or config()
    raw, included = validate_cohort(pd.read_excel(c['master'], sheet_name=c.get('master_sheet', 0)))
    if c.get('npz_dir'):
        require(Path(c['npz_dir']).is_absolute(), 'npz_dir must be absolute')
        included['npz_path'] = included.patient_id.map(lambda p: str(Path(c['npz_dir']) / f'{p}.npz'))
    return raw, included

def identity_digest(included):
    s = included.sort_values('patient_id')[['patient_id', 'model_center', 'severe_mucositis', 'cohort']].to_csv(index=False, lineterminator='\n')
    return hashlib.sha256(s.encode()).hexdigest()

def checked_predictions(frame, expected, probability='probability'):
    frame = frame.copy()
    frame['patient_id'] = integer_values(frame.patient_id, 'prediction patient_id')
    require(not frame.patient_id.duplicated().any(), 'Duplicate prediction patient_id')
    require(set(frame.patient_id) == set(expected.patient_id), 'Prediction IDs do not equal the final cohort')
    frame = frame.set_index('patient_id').loc[expected.patient_id].reset_index()
    y = integer_values(frame.y_true, 'prediction y_true').to_numpy()
    require(np.array_equal(y, expected.severe_mucositis.to_numpy()), 'Prediction outcomes differ from final master')
    p = pd.to_numeric(frame[probability], errors='raise').to_numpy(float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities')
    if 'model_center' in frame:
        require(np.array_equal(frame.model_center.to_numpy(), expected.model_center.to_numpy()), 'Prediction center mismatch')
    frame['probability'] = p
    frame['model_center'] = expected.model_center.to_numpy()
    return frame
