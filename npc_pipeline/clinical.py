from pathlib import Path
from datetime import datetime
import hashlib
import json
import platform
import shutil
import warnings
from itertools import combinations
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels.api as sm
import matplotlib.pyplot as plt
import joblib
from scipy.special import logit
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, brier_score_loss, log_loss, roc_curve
from IPython.display import display
import os
BOOTSTRAP_N = 2000
SEED = 20261023
C_GRID = np.logspace(-3, 3, 13).tolist()
DVH_FEATURES = ['oral_Dmean_Gy', 'oral_V30_percent']
BASE_CONTINUOUS = ['age', 'BMI'] + DVH_FEATURES
BINARY = ['sex', 'smoking', 'drinking', 'induction_chemotherapy', 'concurrent_chemotherapy']
CATEGORICAL = ['T_stage', 'N_stage']
MODEL_SPECS = {'C1_Clinical_Chemo_DVH': BASE_CONTINUOUS, 'C2_Clinical_Chemo_DVH_GTV': BASE_CONTINUOUS + ['log1p_gtv_volume_cm3']}
EXPECTED_COUNTS = {('A', 0): 88, ('A', 1): 92, ('B', 0): 42, ('B', 1): 39, ('C', 0): 116, ('C', 1): 112}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def integer_ids(values, context):
    v = pd.to_numeric(values, errors='raise')
    require(v.notna().all() and np.isfinite(v).all() and (v == np.floor(v)).all(), f'{context}: patient_id含缺失/非整数')
    v = v.astype('int64')
    require(not v.duplicated().any(), f'{context}: patient_id重复')
    return v

def checked_predictions(path, probability_col, reference, expected_count, count_col):
    z = pd.read_csv(path)
    z['patient_id'] = integer_ids(z['patient_id'], str(path))
    z = z.set_index('patient_id')
    require(set(z.index) == set(reference.index), f'{path.name}: 病例集合不一致')
    z = z.loc[reference.index]
    y = pd.to_numeric(z['y_true'], errors='raise')
    require(np.array_equal(y.to_numpy(), reference['severe_mucositis'].to_numpy()), f'{path.name}: 结局不一致')
    p = pd.to_numeric(z[probability_col], errors='raise')
    require(p.notna().all() and np.isfinite(p).all() and p.between(0, 1).all(), f'{path.name}: 概率异常')
    require((pd.to_numeric(z[count_col], errors='raise') == expected_count).all(), f'{path.name}: 预测次数异常')
    if 'model_center' in z:
        require((z['model_center'] == reference['model_center']).all(), f'{path.name}: 中心不一致')
    return p.to_numpy(dtype=float)

def audit_inputs():
    require(MASTER_PATH.is_file(), f'找不到最终master: {MASTER_PATH}')
    raw = pd.read_excel(MASTER_PATH, sheet_name=MASTER_SHEET)
    raw['patient_id'] = integer_ids(raw['patient_id'], 'master')
    require(len(raw) == 540, f'master必须540例，实际{len(raw)}')
    reason = raw['exclude_reason'].fillna('').astype(str).str.strip()
    require(reason.ne('').sum() == 51, '排除人数不等于51')
    require(reason[reason.ne('')].value_counts().to_dict() == {'M1 metastatic disease': 28, 'missing CT RTPLAN': 20, 'no_mask_gtv': 3}, '排除原因计数与冻结队列不符')
    d = raw.loc[reason.eq('')].copy().set_index('patient_id')
    require(len(d) == 489, '纳入人数必须489')
    d['model_center'] = d['model_center'].astype(str).str.strip()
    d['severe_mucositis'] = pd.to_numeric(d['severe_mucositis'], errors='raise')
    require(d['severe_mucositis'].isin([0, 1]).all(), '结局须为0/1')
    d['severe_mucositis'] = d['severe_mucositis'].astype(int)
    require(d.groupby(['model_center', 'severe_mucositis']).size().to_dict() == EXPECTED_COUNTS, '中心×结局计数与冻结489队列不符')
    wanted_cohort = d['model_center'].map({'A': 'External', 'B': 'Development', 'C': 'Development'})
    require((d['cohort'].astype(str).str.strip() == wanted_cohort).all(), 'cohort与正式A外部/B+C开发不符')
    require((d['extraction_status'] == 'PASS').all(), '纳入病例GTV/DVH提取状态存在非PASS')
    d = d.rename(columns={'sex(female=0,male=1)': 'sex', '吸烟史': 'smoking', '饮酒史': 'drinking'})
    columns = list(dict.fromkeys(BASE_CONTINUOUS + BINARY + CATEGORICAL + ['gtv_volume_cm3']))
    for c in columns:
        d[c] = pd.to_numeric(d[c], errors='raise')
        require(d[c].notna().all() and np.isfinite(d[c]).all(), f'{c}有缺失或非有限值；不自动删病例/填0')
    for c in BINARY:
        require(d[c].isin([0, 1]).all(), f'{c}须为0/1')
    require(d['T_stage'].isin([1, 2, 3, 4]).all(), 'T_stage须为1–4')
    require(d['N_stage'].isin([0, 1, 2, 3]).all(), 'N_stage须为0–3')
    require(d['gtv_volume_cm3'].gt(0).all(), 'GTV体积须>0 cm3')
    require(d['age'].between(0, 120, inclusive='neither').all(), '年龄范围异常')
    require(d['BMI'].between(5, 100, inclusive='neither').all(), 'BMI范围异常')
    for c in DVH_FEATURES:
        if c.endswith('_percent'):
            require(d[c].between(0, 100).all(), f'{c}须为0–100百分数，不是比例')
        elif c.endswith('_Gy'):
            require(d[c].between(0, 120).all(), f'{c}单位/范围异常，检查Gy与cGy')
    d['log1p_gtv_volume_cm3'] = np.log1p(d['gtv_volume_cm3'])
    dev = d.loc[d['model_center'].isin(['B', 'C'])].copy()
    ext = d.loc[d['model_center'].eq('A')].copy()
    splits, source_files = ([], [MASTER_PATH])
    for repeat in range(1, 5):
        seen = []
        for fold in range(1, 6):
            folder = SPLIT_ROOT / f'repeat_{repeat:02d}' / f'fold_{fold:02d}'
            item = {'repeat': repeat, 'fold': fold}
            for role in ['outer_train', 'outer_validation', 'inner_train', 'inner_validation']:
                path = folder / f'{role}.csv'
                require(path.is_file(), f'缺固定划分文件: {path}；不自动重划分')
                z = pd.read_csv(path)
                ids = integer_ids(z['patient_id'], str(path)).to_numpy()
                require(set(ids).issubset(set(dev.index)), f'{path}: 出现非当前开发集病例')
                require(np.array_equal(pd.to_numeric(z['severe_mucositis']).to_numpy(), dev.loc[ids, 'severe_mucositis'].to_numpy()), f'{path}: 划分文件与master结局不符')
                require(np.array_equal(z['model_center'].to_numpy(), dev.loc[ids, 'model_center'].to_numpy()), f'{path}: 中心不符')
                require(dev.loc[ids, 'severe_mucositis'].nunique() == 2, f'{path}: 仅有单类结局')
                item[role] = ids
                source_files.append(path)
            ot, ov, it, iv = [set(item[k]) for k in ['outer_train', 'outer_validation', 'inner_train', 'inner_validation']]
            require(not ot & ov and ot | ov == set(dev.index), 'outer划分交叠或未覆盖309人')
            require(not it & iv and it | iv == ot, 'inner划分交叠或未覆盖outer_train')
            require(len(ov) in (61, 62), 'outer_validation人数异常')
            seen.extend(item['outer_validation'])
            splits.append(item)
        require(len(seen) == 309 and len(set(seen)) == 309 and (set(seen) == set(dev.index)), f'repeat {repeat}: 并非每位患者恰有一个外层OOF')
    frozen = {}
    if COMPARE_FROZEN_M3:
        for path in (M3_DEV_PATH, M3_LONG_PATH, M3_EXT_PATH):
            require(path.is_file(), f'缺冻结M3预测: {path}')
        frozen['Development'] = checked_predictions(M3_DEV_PATH, 'oof_probability_mean', dev, 4, 'oof_prediction_count')
        frozen['External'] = checked_predictions(M3_EXT_PATH, 'probability', ext, 20, 'n_checkpoints')
        z = pd.read_csv(M3_LONG_PATH)
        require(len(z) == 1236 and (not z.duplicated(['patient_id', 'repeat']).any()), 'M3长表不是每人4次OOF')
        require(set(z.patient_id) == set(dev.index), 'M3长表病例集合不一致')
        require(set(zip(z['repeat'], z['fold'])) == {(r, f) for r in range(1, 5) for f in range(1, 6)}, 'M3长表repeat/fold异常')
        require(np.isfinite(z.probability).all() and z.probability.between(0, 1).all(), 'M3长表概率异常')
        require(np.array_equal(z.y_true.to_numpy(), dev.loc[z.patient_id, 'severe_mucositis'].to_numpy()), 'M3长表结局不符')
        for split in splits:
            part = z.loc[(z['repeat'] == split['repeat']) & (z['fold'] == split['fold'])]
            require(set(part.patient_id) == set(split['outer_validation']), 'M3预测与冻结outer_validation ID不一致')
        means = z.groupby('patient_id').probability.mean().reindex(dev.index).to_numpy()
        require(np.allclose(means, frozen['Development'], atol=1e-10, rtol=1e-08), 'M3均值与长表不一致')
        source_files += [M3_DEV_PATH, M3_LONG_PATH, M3_EXT_PATH]
    print('输入审计通过：540 assessed / 51 excluded / 489 included; B+C=309, A=180。')
    print('20 fixed inner/outer splits verified against final master.')
    print('M3冻结预测及OOF划分核对:', '通过' if COMPARE_FROZEN_M3 else '未启用')
    return (dev, ext, splits, frozen, source_files)

def make_pipeline(continuous):
    prep = ColumnTransformer([('continuous', StandardScaler(), continuous), ('binary', 'passthrough', BINARY), ('stage', OneHotEncoder(categories=[[1, 2, 3, 4], [0, 1, 2, 3]], drop='first', handle_unknown='error', sparse_output=False), CATEGORICAL)], remainder='drop', verbose_feature_names_out=True)
    return Pipeline([('prepare', prep), ('lr', LogisticRegression(penalty='l2', solver='lbfgs', max_iter=10000, tol=1e-07, class_weight=None, random_state=SEED))])

def save_csv(frame, path):
    frame.to_csv(path, index=False, encoding='utf-8-sig', float_format='%.12g')

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')

def make_output_folder(files):
    output = OUTPUT_ROOT / ('Clinical_DVH_GTV_comparators_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir()
    (output / 'input_snapshot').mkdir()
    hashes = []
    for path in files:
        path = Path(path)
        rel = Path(hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]) / path.name
        target = output / 'input_snapshot' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        original_hash = sha256_file(path)
        shutil.copy2(path, target)
        require(sha256_file(target) == original_hash, f'输入快照复制不一致: {path}')
        hashes.append({'source': str(path), 'snapshot': str(target.relative_to(output)), 'sha256': original_hash})
    save_csv(pd.DataFrame(hashes), output / 'input_hashes.csv')
    config = {'analysis': 'Revision-added exploratory treatment-background comparators', 'models': {k: v + BINARY + CATEGORICAL for k, v in MODEL_SPECS.items()}, 'C_grid': C_GRID, 'tuning_metric': 'inner_validation log loss', 'cohort': {'assessed': 540, 'excluded': 51, 'development': 309, 'external': 180}, 'split_protocol': 'Existing 4x5 outer and inner splits, refit complete outer_train', 'PR_AUC': 'trapezoidal auc(recall, precision)', 'bootstrap': {'n': BOOTSTRAP_N, 'seed': SEED, 'unit': 'patient', 'retrain': False, 'method': 'unstratified paired percentile; reject single-outcome replicates'}, 'concurrent_chemo': 'actually received, not necessarily available at prediction time', 'compare_frozen_M3': COMPARE_FROZEN_M3, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'statsmodels': sm.__version__}, 'created_at': datetime.now().isoformat()}
    save_json(config, output / 'analysis_config.json')
    return output

def train_comparators(dev, splits, output):
    records, selections, coefficients, ensembles = ([], [], [], {})
    for model, continuous in MODEL_SPECS.items():
        ensembles[model] = []
        feature_columns = continuous + BINARY + CATEGORICAL
        for split in splits:
            r, f = (split['repeat'], split['fold'])
            train_ids, val_ids = (split['outer_train'], split['outer_validation'])
            X = dev.loc[train_ids, feature_columns]
            y = dev.loc[train_ids, 'severe_mucositis']
            position = {pid: i for i, pid in enumerate(train_ids)}
            it = np.array([position[pid] for pid in split['inner_train']])
            iv = np.array([position[pid] for pid in split['inner_validation']])
            search = GridSearchCV(make_pipeline(continuous), {'lr__C': C_GRID}, scoring='neg_log_loss', cv=[(it, iv)], refit=True, n_jobs=1, error_score='raise', return_train_score=False)
            with warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                search.fit(X, y)
            fitted = search.best_estimator_
            probability = fitted.predict_proba(dev.loc[val_ids, feature_columns])[:, 1]
            rec = pd.DataFrame({'patient_id': val_ids, 'y_true': dev.loc[val_ids, 'severe_mucositis'].to_numpy(), 'model_center': dev.loc[val_ids, 'model_center'].to_numpy(), 'model': model, 'repeat': r, 'fold': f, 'probability': probability})
            records.append(rec)
            name = f'{model}_r{r:02d}_f{f:02d}'
            joblib.dump(fitted, output / 'models' / f'{name}.joblib')
            ensemble_item = {'pipeline': fitted, 'repeat': r, 'fold': f, 'features': feature_columns}
            ensembles[model].append(ensemble_item)
            save_csv(pd.DataFrame(search.cv_results_), output / 'models' / f'{name}_C_search.csv')
            selected_C = float(search.best_params_['lr__C'])
            selections.append({'model': model, 'repeat': r, 'fold': f, 'C': selected_C, 'inner_log_loss': float(-search.best_score_), 'C_at_grid_edge': selected_C in (C_GRID[0], C_GRID[-1]), 'outer_train_n': len(train_ids), 'outer_validation_n': len(val_ids), 'inner_train_n': len(it), 'inner_validation_n': len(iv)})
            names = fitted.named_steps['prepare'].get_feature_names_out().tolist()
            vals = fitted.named_steps['lr'].coef_[0].tolist()
            for feature, coef in zip(['intercept'] + names, [float(fitted.named_steps['lr'].intercept_[0])] + vals):
                coefficients.append({'model': model, 'repeat': r, 'fold': f, 'feature': feature, 'coefficient': coef})
            print(f'{model}: repeat {r}/4, fold {f}/5, selected C={selected_C:g}', flush=True)
    long = pd.concat(records, ignore_index=True)
    require(len(long) == 2 * 1236, 'OOF总记录数异常')
    require(not long.duplicated(['model', 'patient_id', 'repeat']).any(), 'OOF重复')
    require(long.groupby(['model', 'patient_id']).size().eq(4).all(), 'OOF必须每人4次')
    save_csv(long, output / 'comparator_oof_long.csv')
    save_csv(pd.DataFrame(selections), output / 'selected_C.csv')
    save_csv(pd.DataFrame(coefficients), output / 'coefficients_standardized.csv')
    model_hashes = [{'file': str(p.relative_to(output)), 'sha256': sha256_file(p)} for p in sorted((output / 'models').glob('*.joblib'))]
    require(len(model_hashes) == 40, '应有2×20个冻结模型')
    save_json({'frozen_at': datetime.now().isoformat(), 'files': model_hashes}, output / 'MODELS_FROZEN_BEFORE_EXTERNAL.json')
    return (ensembles, long)

def assemble_predictions(dev, ext, ensembles, oof_long, frozen, output):
    tables = {}
    for cohort, reference in [('Development', dev), ('External', ext)]:
        z = reference[['severe_mucositis', 'model_center']].rename(columns={'severe_mucositis': 'y_true'}).copy()
        z['cohort'] = cohort
        tables[cohort] = z
    external_long = []
    for model, items in ensembles.items():
        means = oof_long.loc[oof_long.model.eq(model)].groupby('patient_id').probability.mean()
        tables['Development'][model] = means.reindex(dev.index)
        require(len(items) == 20, '每模型必须冻结20折')
        predictions = []
        for item in items:
            p = item['pipeline'].predict_proba(ext[item['features']])[:, 1]
            predictions.append(p)
            external_long.append(pd.DataFrame({'patient_id': ext.index, 'model': model, 'repeat': item['repeat'], 'fold': item['fold'], 'probability': p}))
        tables['External'][model] = np.mean(predictions, axis=0)
    for cohort, table in tables.items():
        if frozen:
            table['M3_CNN_frozen'] = frozen[cohort]
        require(table.notna().all().all(), f'{cohort}: 合并预测出现缺失')
        save_csv(table.reset_index(), output / f'{cohort}_patient_predictions.csv')
    save_csv(pd.concat(external_long, ignore_index=True), output / 'comparator_external_20model_long.csv')
    return tables

def metric_vector(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), '预测概率范围异常')
    precision, recall, _ = precision_recall_curve(y, p)
    return np.array([roc_auc_score(y, p), auc(recall, precision), brier_score_loss(y, p), log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])])

def bootstrap_metrics(table, models, n_boot, seed):
    require(n_boot >= 100, '正式bootstrap至少100次，默认2000；不要将测试次数作为正式区间')
    y = table.y_true.to_numpy(dtype=int)
    probs = table[models].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(models), len(METRICS)))
    for b in range(n_boot):
        while True:
            idx = rng.integers(0, len(y), len(y))
            if np.unique(y[idx]).size == 2:
                break
        for j in range(len(models)):
            draws[b, j] = metric_vector(y[idx], probs[idx, j])
        if (b + 1) % 500 == 0:
            print(f'  bootstrap {b + 1}/{n_boot}', flush=True)
    points = np.stack([metric_vector(y, probs[:, j]) for j in range(len(models))])
    rows, delta = ([], [])
    for j, model in enumerate(models):
        lo, hi = np.quantile(draws[:, j], [0.025, 0.975], axis=0)
        for k, metric in enumerate(METRICS):
            rows.append({'model': model, 'metric': metric, 'estimate': points[j, k], 'ci_low': lo[k], 'ci_high': hi[k], 'n': len(y), 'events': int(y.sum()), 'bootstrap_n': n_boot})
    for ia, ib in combinations(range(len(models)), 2):
        lo, hi = np.quantile(draws[:, ib] - draws[:, ia], [0.025, 0.975], axis=0)
        for k, metric in enumerate(METRICS):
            delta.append({'model_A': models[ib], 'model_B': models[ia], 'metric': metric, 'difference_A_minus_B': points[ib, k] - points[ia, k], 'ci_low': lo[k], 'ci_high': hi[k], 'bootstrap_n': n_boot})
    return (pd.DataFrame(rows), pd.DataFrame(delta))

def calibration_stats(y, p):
    y = np.asarray(y, dtype=int)
    lp = logit(np.clip(p, 1e-06, 1 - 1e-06))
    result = {'CITL': np.nan, 'calibration_intercept_joint': np.nan, 'calibration_slope': np.nan, 'calibration_fit_status': 'OK'}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            intercept_fit = sm.GLM(y, np.ones((len(y), 1)), offset=lp, family=sm.families.Binomial()).fit(maxiter=200)
            slope_fit = sm.GLM(y, sm.add_constant(lp, has_constant='add'), family=sm.families.Binomial()).fit(maxiter=200)
        require(intercept_fit.converged and slope_fit.converged, '校准回归未收敛')
        require(np.isfinite(intercept_fit.params).all() and np.isfinite(slope_fit.params).all(), '校准回归参数非有限值')
        result.update(CITL=float(intercept_fit.params[0]), calibration_intercept_joint=float(slope_fit.params[0]), calibration_slope=float(slope_fit.params[1]))
    except (Exception,) as e:
        result['calibration_fit_status'] = 'FAILED: ' + str(e)
        warnings.warn(result['calibration_fit_status'])
    return result

def calibration_groups(table, model):
    z = table.reset_index()[['patient_id', 'y_true', model]].sort_values([model, 'patient_id'])
    rows = []
    zz = norm.ppf(0.975)
    for group, idx in enumerate(np.array_split(np.arange(len(z)), 10), start=1):
        part = z.iloc[idx]
        n, events = (len(part), int(part.y_true.sum()))
        observed = events / n
        denominator = 1 + zz ** 2 / n
        center = (observed + zz ** 2 / (2 * n)) / denominator
        half = zz * np.sqrt(observed * (1 - observed) / n + zz ** 2 / (4 * n * n)) / denominator
        rows.append({'group': group, 'n': n, 'events': events, 'mean_predicted': part[model].mean(), 'observed_rate': observed, 'wilson_low': max(0, center - half), 'wilson_high': min(1, center + half)})
    return pd.DataFrame(rows)

def dca_curve(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p)
    rows = []
    for threshold in np.arange(1, 100) / 100:
        positive = p >= threshold
        tp = np.sum(positive & (y == 1))
        fp = np.sum(positive & (y == 0))
        odds = threshold / (1 - threshold)
        rows.append({'threshold': threshold, 'net_benefit': tp / len(y) - fp / len(y) * odds, 'treat_all': y.mean() - (1 - y.mean()) * odds, 'treat_none': 0.0})
    return pd.DataFrame(rows)

def evaluate_predictions(tables, output):
    all_metrics, all_deltas, all_calibration, all_groups, all_dca, descriptive = ([], [], [], [], [], [])
    models = list(MODEL_SPECS) + (['M3_CNN_frozen'] if COMPARE_FROZEN_M3 else [])
    for cohort_no, (cohort, table) in enumerate(tables.items()):
        print(f'{cohort}: {len(table)} patients; evaluating frozen predictions...', flush=True)
        metrics, differences = bootstrap_metrics(table, models, BOOTSTRAP_N, SEED + cohort_no)
        all_metrics.append(metrics.assign(cohort=cohort))
        all_deltas.append(differences.assign(cohort=cohort))
        y = table.y_true.to_numpy(dtype=int)
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), layout='constrained')
        for model in models:
            p = table[model].to_numpy(dtype=float)
            all_calibration.append(dict(cohort=cohort, model=model, **calibration_stats(y, p)))
            groups = calibration_groups(table, model)
            all_groups.append(groups.assign(cohort=cohort, model=model))
            curve = dca_curve(y, p)
            all_dca.append(curve.assign(cohort=cohort, model=model))
            positive = p >= 0.5
            descriptive.append({'cohort': cohort, 'model': model, 'threshold': 0.5, 'sensitivity': float(positive[y == 1].mean()), 'specificity': float((~positive[y == 0]).mean()), 'accuracy': float((positive == y).mean())})
            fpr, tpr, _ = roc_curve(y, p)
            axes[0].plot(fpr, tpr, label=model)
            precision, recall, _ = precision_recall_curve(y, p)
            axes[1].plot(recall, precision, label=model)
            axes[2].plot(groups.mean_predicted, groups.observed_rate, marker='o', label=model)
        axes[0].plot([0, 1], [0, 1], '--', color='gray')
        axes[1].axhline(y.mean(), ls='--', color='gray')
        axes[2].plot([0, 1], [0, 1], '--', color='gray')
        for ax, xlabel, ylabel, title in zip(axes, ['False positive rate', 'Recall', 'Mean predicted risk'], ['True positive rate', 'Precision', 'Observed risk'], ['ROC', 'Precision–recall', 'Grouped calibration']):
            ax.set(xlabel=xlabel, ylabel=ylabel, title=cohort + ': ' + title, xlim=(0, 1), ylim=(0, 1))
        axes[0].legend(fontsize=7, loc='lower right')
        fig.savefig(output / f'{cohort}_diagnostic_curves.png', dpi=180)
        plt.close(fig)
    outputs = {'metrics_with_95CI.csv': pd.concat(all_metrics, ignore_index=True), 'paired_differences_95CI.csv': pd.concat(all_deltas, ignore_index=True), 'calibration_summary.csv': pd.DataFrame(all_calibration), 'calibration_10groups_Wilson.csv': pd.concat(all_groups, ignore_index=True), 'DCA_thresholds_001_099.csv': pd.concat(all_dca, ignore_index=True), 'threshold_050_descriptive.csv': pd.DataFrame(descriptive)}
    for filename, frame in outputs.items():
        save_csv(frame, output / filename)
    return outputs
def configure():
    global ROOT, OUTPUT_ROOT, MASTER_PATH, MASTER_SHEET, SPLIT_ROOT, M3_DEV_PATH, M3_LONG_PATH, M3_EXT_PATH, COMPARE_FROZEN_M3
    from .common import config, load_cohort
    from .splits import load_locked
    c=config(); load_cohort(c); load_locked(c)
    ROOT=OUTPUT_ROOT=Path(c['result_root']); OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    MASTER_PATH=Path(c['master']); MASTER_SHEET=c.get('master_sheet',0); SPLIT_ROOT=Path(c['split_dir'])
    COMPARE_FROZEN_M3=bool(c.get('compare_frozen_m3', True))
    prediction=Path(c.get('prediction_dir', c['result_root']))
    M3_DEV_PATH=prediction/'development/M3_Oral_GTV_MaskedDose_patient_averaged_oof_309.csv'
    M3_LONG_PATH=prediction/'development/M3_Oral_GTV_MaskedDose_oof_long_1236.csv'
    M3_EXT_PATH=prediction/'external/M3_Oral_GTV_MaskedDose_CenterA_180_20model_ensemble_predictions.csv'

def run():
    from .common import register
    configure()
    dev,ext,splits,frozen_m3,source_files=audit_inputs()
    output=make_output_folder(source_files)
    print('Output:',output,flush=True)
    ensembles,oof=train_comparators(dev,splits,output)
    tables=assemble_predictions(dev,ext,ensembles,oof,frozen_m3,output)
    evaluate_predictions(tables,output)
    for path in source_files:
        snapshot=output/'input_snapshot'/Path(hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12])/path.name
        require(sha256_file(path)==sha256_file(snapshot),'Input changed during run')
    save_json({'status':'COMPLETED','development_n':309,'external_n':180,'models_per_comparator':20},output/'RUN_COMPLETED.json')
    register('clinical_run',str(output))
    return output
