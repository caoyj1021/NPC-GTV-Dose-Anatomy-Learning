"""Bind downstream calibration and attribution artifacts to their actual inputs."""
from pathlib import Path
import json
from .common import config, sha256, require, write_json


def context(stage):
    c = config()
    result = {
        'master_sha256': sha256(c['master']),
        'prediction_audit_sha256': sha256(Path(c['prediction_dir']) / 'prediction_audit.json'),
        'split_lock_sha256': sha256(Path(c['split_dir']) / 'SPLITS_LOCKED.json'),
    }
    if stage == 'gradcam':
        result['npz_dir'] = str(Path(c['npz_dir']).resolve())
    return result


def lock_artifact(folder, stage, input_paths=()):
    folder = Path(folder)
    outputs = {str(p.relative_to(folder)): sha256(p) for p in folder.rglob('*')
               if p.is_file() and p.name != 'ARTIFACT_LOCK.json'}
    require(outputs, 'Cannot lock an empty artifact directory')
    write_json(folder / 'ARTIFACT_LOCK.json', {
        'stage': stage, 'context': context(stage), 'outputs': outputs,
        'inputs': {str(Path(p).resolve()): sha256(p) for p in input_paths},
    })


def verify_artifact(folder, stage):
    folder = Path(folder)
    marker = folder / 'ARTIFACT_LOCK.json'
    require(marker.is_file(), f'Missing {stage} artifact lock; regenerate its upstream analysis')
    lock = json.loads(marker.read_text(encoding='utf-8'))
    require(lock['stage'] == stage and lock['context'] == context(stage),
            f'{stage} belongs to different inputs; regenerate its upstream analysis')
    for relative, digest in lock['outputs'].items():
        p = (folder / relative).resolve()
        require(p.is_relative_to(folder.resolve()) and p.is_file() and sha256(p) == digest,
                f'{stage} output missing or changed: {relative}')
    for name, digest in lock['inputs'].items():
        p = Path(name)
        require(p.is_file() and sha256(p) == digest, f'{stage} source missing or changed: {name}')
    return lock
