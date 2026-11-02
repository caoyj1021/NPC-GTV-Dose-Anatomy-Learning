"""Frozen-prediction paired analysis; no fitting, tuning, or source edits."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json, hashlib, platform
import numpy as np
import pandas as pd
import os
from .common import new_output, register
ROOT = None
OUT = None
B, SEED = (2000, 20261025)
METRICS = ['ROC_AUC', 'PR_AUC_trapezoidal', 'Brier', 'Log_loss']
MODELS = [('M1', 1, 'D', 'M1_Dose_Only'), ('M2_Abl', 2, 'D + O', 'M2_Abl'), ('DoseGTV', 2, 'D + G', None), ('MasksOnly', 2, 'O + G', None), ('M2', 3, 'D + O + D×O', 'M2_Oral_MaskedDose'), ('M3_Abl', 3, 'D + O + G', 'M3_Abl'), ('M3', 5, 'D + O + G + D×O + D×G', 'M3_Oral_GTV_MaskedDose'), ('M4', 6, 'CT + D + O + G + D×O + D×G', 'M4_CT_Augmented_M3'), ('C1', None, '临床因素 + 化疗标签 + 口腔DVH', None), ('C2', None, 'C1 + log(1+GTV体积)', None)]
PAIRS = [('P01', 'M2_Abl', 'M1', '在D基础上加入O', '显式口腔mask的增益'), ('P02', 'DoseGTV', 'M1', '在D基础上加入G', '显式GTV mask的增益'), ('P03', 'M3_Abl', 'M2_Abl', '在D+O基础上加入G', '已有剂量及口腔信息时GTV的增益'), ('P04', 'M3_Abl', 'DoseGTV', '在D+G基础上加入O', '已有剂量及GTV时口腔mask的增益'), ('P05', 'M3_Abl', 'MasksOnly', '在O+G基础上加入D', '原始剂量通道相对于纯解剖输入的增益'), ('P06', 'M2', 'M2_Abl', '在D+O基础上加入D×O', '口腔条件剂量通道的增益'), ('P07', 'M3', 'M3_Abl', '同时加入D×O和D×G', '结构条件剂量通道的联合增益'), ('P08', 'M3', 'MasksOnly', '同时加入D、D×O和D×G', '完整剂量输入组合相对于纯解剖模型'), ('P09', 'M4', 'M3', '在完整M3基础上加入CT', 'CT的额外增益'), ('P10', 'M3', 'DoseGTV', '同时加入O、D×O和D×G', '完整模型相对于D+G精简配置'), ('P11', 'C2', 'C1', '加入log(1+GTV体积)', '单一GTV体积变量的增益'), ('P12', 'M3', 'C1', '完整CNN与临床化疗DVH对照', '仅对本研究指定传统对照的比较'), ('P13', 'M3', 'C2', '完整CNN与含GTV体积的传统对照', '不能单独归因于GTV空间关系')]

def metrics(y, p):
    """Exact empirical tie-aware ROC AUC and trapezoidal PR AUC."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    order = np.argsort(-p, kind='stable')
    ys, ps = (y[order], p[order])
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(y) - 1]
    tp = np.cumsum(ys)[ends].astype(float)
    fp = ends + 1 - tp
    positives = y.sum()
    negatives = len(y) - positives
    if not (positives > 0 and negatives > 0):
        raise ValueError('Both outcomes required')
    roc = np.trapezoid(np.r_[0, tp / positives], np.r_[0, fp / negatives])
    pr = np.trapezoid(np.r_[1, tp / (tp + fp)], np.r_[0, tp / positives])
    clipped = np.clip(p, np.finfo(float).eps, 1 - np.finfo(float).eps)
    ll = -np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    return np.array([roc, pr, np.mean((y - p) ** 2), ll])

def main():
    global OUT
    OUT = new_output('All_Model_Comparisons')
    sources, audit_rows, predictions, stats, paired = ([], [], {}, [], [])
    seen = set()

    def record(path):
        if str(path) not in seen:
            sources.append({'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            seen.add(str(path))

    def load(path):
        record(path)
        d = pd.read_csv(path)
        assert d.patient_id.notna().all() and (not d.patient_id.duplicated().any())
        assert np.equal(d.patient_id, np.floor(d.patient_id)).all()
        return d.set_index('patient_id').sort_index()
    from .predictions import load_aligned
    tables = load_aligned(require_all=True)
    from .common import config
    for cohort, reference_frame in tables.items():
        path = Path(config()['prediction_dir']) / f'{cohort}_aligned_predictions.csv'
        record(path)
        reference = reference_frame.set_index('patient_id').sort_index()
        n = len(reference)
        events = int(reference.y_true.sum())
        y = reference.y_true.to_numpy(dtype=int)
        mat = np.array([reference[name].to_numpy(dtype=float) for name, _, _, _ in MODELS])
        for name, _, _, _ in MODELS:
            audit_rows.append({'cohort': cohort, 'model': name, 'n': n, 'events': events, 'unique_ids': n, 'labels_match': True, 'source_file': path.name})
        values = np.array([metrics(y, p) for p in mat])
        rng = np.random.default_rng(SEED)
        draws = np.empty((B, len(MODELS), 4))
        indices = []
        rejected = 0
        for b in range(B):
            while True:
                ix = rng.integers(0, n, n)
                if len(np.unique(y[ix])) == 2:
                    break
                rejected += 1
            indices.append(ix)
            for j, p in enumerate(mat):
                draws[b, j] = metrics(y[ix], p[ix])
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        for j, (name, channels, inputs, _) in enumerate(MODELS):
            for k, metric in enumerate(METRICS):
                stats.append({'cohort': cohort, 'model': name, 'channels': channels, 'inputs': inputs, 'n': n, 'events': events, 'metric': metric, 'estimate': float(values[j, k]), 'ci_low': float(lo[j, k]), 'ci_high': float(hi[j, k])})
        names = [m[0] for m in MODELS]
        for pid, a, b, change, purpose in PAIRS:
            ja, jb = (names.index(a), names.index(b))
            diff = values[ja] - values[jb]
            lower, upper = np.percentile(draws[:, ja] - draws[:, jb], [2.5, 97.5], axis=0)
            for k, metric in enumerate(METRICS):
                favors_a = lower[k] > 0 if k < 2 else upper[k] < 0
                favors_b = upper[k] < 0 if k < 2 else lower[k] > 0
                conclusion = '区间跨0，方向未明确' if not (favors_a or favors_b) else '区间不跨0，支持A较好' if favors_a else '区间不跨0，支持B较好'
                paired.append({'cohort': cohort, 'comparison': pid, 'model_A': a, 'model_B': b, 'change': change, 'purpose': purpose, 'metric': metric, 'A_estimate': float(values[ja, k]), 'B_estimate': float(values[jb, k]), 'difference_A_minus_B': float(diff[k]), 'ci_low': float(lower[k]), 'ci_high': float(upper[k]), 'CI_interpretation_unadjusted': conclusion})
        pred = pd.DataFrame({'patient_id': reference.index, 'y_true': y})
        for j, name in enumerate(names):
            pred[name] = mat[j]
        pred.to_csv(OUT / f'{cohort}_aligned_predictions.csv', index=False, encoding='utf-8-sig')
        np.savez_compressed(OUT / f'{cohort}_bootstrap_draws.npz', metrics=draws, patient_indices=np.asarray(indices))
        predictions[cohort] = pred.to_dict(orient='records')
        print(cohort, 'PASS', n, 'patients;', B, 'shared paired resamples; rejected', rejected, flush=True)
    stat_df = pd.DataFrame(stats)
    output = {'created_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'method': {'bootstrap_n': B, 'seed': SEED, 'resampling': 'Ordinary patient bootstrap within each cohort; identical indices for every model; single-class samples rejected.', 'CI': '2.5th and 97.5th percentiles; pointwise, unadjusted for multiplicity.', 'scope': 'Exploratory comparisons selected by scientific questions after some point estimates were observed. Not preregistered.', 'limitations': 'Frozen-prediction uncertainty only. No refitting; does not capture training uncertainty. Development OOF observations share training data.', 'PR_AUC': 'Trapezoidal area, with recall on x-axis. Not average precision.', 'direction': 'All differences A minus B. Higher AUC/PR-AUC and lower Brier/Log loss are better.', 'software': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__}}, 'models': [{'id': m[0], 'channels': m[1], 'inputs': m[2]} for m in MODELS], 'comparisons': [dict(zip(['id', 'A', 'B', 'change', 'purpose'], p)) for p in PAIRS], 'performance': stats, 'paired': paired, 'audit': audit_rows, 'sources': sources}
    (OUT / 'analysis.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    stat_df.to_csv(OUT / 'model_performance_95CI.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(paired).to_csv(OUT / 'paired_differences_95CI.csv', index=False, encoding='utf-8-sig')
    print(pd.DataFrame(paired).query("cohort=='External' and metric in ['ROC_AUC','Brier']")[['comparison', 'model_A', 'model_B', 'metric', 'difference_A_minus_B', 'ci_low', 'ci_high', 'CI_interpretation_unadjusted']].to_string(index=False))
    register('paired_analysis_dir', str(OUT))
    print('SAVED', OUT)
    return OUT
if __name__ == '__main__':
    main()
