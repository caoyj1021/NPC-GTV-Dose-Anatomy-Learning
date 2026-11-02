"""Public execution wrappers; numerical training core is shared by all CNNs."""
from pathlib import Path
import json
import os
import traceback
import numpy as np
import pandas as pd
from .common import config, load_cohort, require, sha256, identity_digest, register, write_json
from .splits import load_locked
from . import cnn_engine as engine

MODELS = {
    'M1_Dose_Only': ['dose'],
    'M2_Abl': ['dose', 'oral'],
    'M2_Oral_MaskedDose': ['dose', 'oral', 'dose_oral'],
    'M3_Abl': ['dose', 'oral', 'gtv'],
    'M3_Oral_GTV_MaskedDose': ['dose', 'oral', 'gtv', 'dose_oral', 'dose_gtv'],
    'M4_CT_Augmented_M3': ['ct', 'dose', 'oral', 'gtv', 'dose_oral', 'dose_gtv'],
    'R1A_MasksOnly': ['oral', 'gtv'],
    'R1B_DoseGTV': ['dose', 'gtv'],
}
ORIGINAL_MODELS = list(MODELS)[:6]
REVISION_MODELS = list(MODELS)[6:]

def configure(model):
    c = config()
    require(model in MODELS, 'Unknown CNN model')
    engine.ROOT = Path(c['result_root'])
    engine.ROOT.mkdir(parents=True, exist_ok=True)
    engine.MASTER_PATH = Path(c['master'])
    engine.MASTER_SHEET = c.get('master_sheet', 0)
    engine.NPZ_DIR = Path(c['npz_dir'])
    engine.MODEL_NAME, engine.CHANNELS = model, MODELS[model]
    engine.RESUME_RUN_DIR = c.get('resume_cnn_runs', {}).get(model)
    engine.SOURCE_BUILD = sha256(Path(engine.__file__)) + ':' + sha256(Path(__file__))
    engine.preflight = preflight
    return c

def preflight():
    c = config()
    raw, included = load_cohort(c)
    frames, configurations, input_paths = load_locked(c)
    input_paths['master.xlsx'] = Path(c['master'])
    input_paths['SPLITS_LOCKED.json'] = Path(c['split_dir']) / 'SPLITS_LOCKED.json'
    for parts in frames.values():
        for role, frame in parts.items():
            parts[role] = frame.sort_values('patient_id').reset_index(drop=True)
    dev = included.loc[included.model_center.isin(['B', 'C'])].copy()
    ext = included.loc[included.model_center.eq('A')].copy()
    hashes = []
    for row in included.itertuples():
        path = Path(row.npz_path)
        require(path.is_file(), f'Missing NPZ: {path}')
        hashes.append({'patient_id': int(row.patient_id), 'path': str(path), 'sha256': sha256(path)})
    dataset = engine.NPCNPZDataset(dev, engine.CHANNELS, training=False)
    for i in range(len(dataset)):
        x, _, _ = dataset[i]
        require(tuple(x.shape) == (len(engine.CHANNELS), *engine.EXPECTED_SHAPE_ZYX), 'Input shape mismatch')
    inputs = [{'key': k, 'path': str(p), 'sha256': sha256(p)} for k, p in sorted(input_paths.items())]
    payload = {'model': engine.MODEL_NAME, 'channels': engine.CHANNELS,
        'source_build': engine.SOURCE_BUILD, 'training_settings': engine.training_settings(),
        'identity_digest': identity_digest(included), 'inputs': inputs, 'npz_hashes': hashes,
        'n_development': 309, 'n_external': 180, 'precision_recall_auc': 'trapezoidal'}
    import hashlib
    signature = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return dict(raw=raw, included=included, dev=dev, ext=ext, frames=frames, configurations=configurations,
                input_paths=input_paths, payload=payload, signature=signature)

def train(models=None):
    """Train and freeze every requested model. This stage does not evaluate external performance."""
    models = models or ORIGINAL_MODELS
    require(engine.torch.cuda.is_available(), 'Full training requires a CUDA PyTorch environment')
    for model in models:
        configure(model)
        audit = preflight()
        folder = engine.initialize_run(audit)
        print('Run directory:', folder, flush=True)
        lock = folder / 'RUNNING.lock'
        with lock.open('x') as stream:
            stream.write(str(os.getpid()))
        try:
            write_json(folder / 'environment.json', engine.environment_record())
            records = engine.train_tasks(folder, audit, engine.torch.device('cuda'))
            dev_predictions = engine.aggregate_development(folder, audit, records)
            # Freeze before the separate external-validation stage.
            frozen = {'run_signature': audit['signature'], 'model': model,
                'identity_digest': identity_digest(audit['included']), 'records': records,
                'development_youden': engine.development_youden_threshold(dev_predictions.y_true, dev_predictions.probability),
                'development_prediction_sha256': sha256(folder / 'aggregate/Development_patient_predictions_309.csv'),
                'checkpoint_hashes': {r['checkpoint']: sha256(folder / r['checkpoint']) for r in records}}
            write_json(folder / 'TRAINING_FROZEN.json', frozen)
            runs = config().get('cnn_runs', {})
            runs[model] = str(folder)
            register('cnn_runs', runs)
        except BaseException:
            (folder / 'FAILED.txt').write_text(traceback.format_exc(), encoding='utf-8')
            raise
        finally:
            lock.unlink(missing_ok=True)
    print('Development training frozen. Run the external-validation notebook next.')
    return config()['cnn_runs']

def validate(models=None):
    c = config()
    models = models or list(c.get('cnn_runs', {}))
    require(models, 'No frozen model runs configured')
    frozen_runs = {}
    _, included = load_cohort(c)
    # Verify the entire requested set before loading external images.
    for model in models:
        folder = Path(c['cnn_runs'][model])
        frozen = json.loads((folder / 'TRAINING_FROZEN.json').read_text(encoding='utf-8'))
        require(frozen['model'] == model and len(frozen['records']) == 20, 'Wrong/incomplete frozen model')
        require(frozen['identity_digest'] == identity_digest(included), 'Frozen cohort mismatch')
        require({(r['repeat'], r['fold']) for r in frozen['records']} == {(r, f) for r in range(1, 5) for f in range(1, 6)}, 'Incomplete fold set')
        for rel, digest in frozen['checkpoint_hashes'].items():
            require(sha256(folder / rel) == digest, 'Checkpoint changed after freezing')
        require(sha256(folder / 'aggregate/Development_patient_predictions_309.csv') == frozen['development_prediction_sha256'], 'Development predictions changed after freezing')
        for r in frozen['records']:
            engine.task_record(folder, (r['repeat'], r['fold']), frozen['run_signature'])
        frozen_runs[model] = (folder, frozen)
    for model, (folder, frozen) in frozen_runs.items():
        configure(model)
        audit = preflight()
        require(audit['signature'] == frozen['run_signature'], 'Inputs or code changed since training')
        if (folder / 'RUN_COMPLETED.json').exists():
            print('Completed external validation already exists:', folder)
            continue
        dev = pd.read_csv(folder / 'aggregate/Development_patient_predictions_309.csv')
        engine.validate_predictions(dev, audit['dev'])
        device = engine.torch.device('cuda' if engine.torch.cuda.is_available() else 'cpu')
        ext = engine.infer_external(folder, audit, frozen['records'], device)
        engine.summarize_results(folder, audit, dev, ext)
        write_json(folder / 'RUN_COMPLETED.json', {'status': 'PASS', 'model': model, 'channels': MODELS[model],
            'n_models': 20, 'n_development': 309, 'n_external': 180, 'run_signature': audit['signature']})
    return {k: str(v[0]) for k, v in frozen_runs.items()}
