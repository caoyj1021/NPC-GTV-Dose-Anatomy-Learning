from pathlib import Path
import sys, json, hashlib, time
import os
from datetime import datetime
from .common import config, new_output, load_cohort, register
from .predictions import load_aligned
c = config()
load_cohort(c)
load_aligned(require_all=True)
OUT = new_output('Additional_Analyses')
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(c['result_root'])
master = Path(c['master'])
snapshot = OUT / 'master_snapshot.xlsx'
import shutil
shutil.copy2(master, snapshot)
source = Path(c['prediction_dir'])
qcfile = Path(c['table2_qc']) if c.get('table2_qc') else None
cols = ['patient_id', 'model_center', 'cohort', 'RT_start_year', 'severe_mucositis', 'gtv_volume_cm3']
raw = pd.read_excel(master, sheet_name=c.get('master_sheet', 0))
snap = pd.read_excel(snapshot, sheet_name=c.get('master_sheet', 0))
pd.testing.assert_frame_equal(raw[cols], snap[cols])
m = raw.loc[raw.exclude_reason.fillna('').astype(str).str.strip().eq(''), cols].copy()
assert len(raw) == 540 and len(m) == 489 and (not m.patient_id.duplicated().any())
assert m[cols].notna().all().all()
m.patient_id = m.patient_id.astype(int)
m.RT_start_year = m.RT_start_year.astype(int)
assert m.gtv_volume_cm3.gt(0).all()
blocks = []
for cohort, n in [('Development', 309), ('External', 180)]:
    a = pd.read_csv(source / f'{cohort}_aligned_predictions.csv')
    a = a.drop(columns=['model_center'], errors='ignore').merge(m, on='patient_id', how='left', validate='one_to_one')
    assert len(a) == n and a[cols].notna().all().all()
    assert (a.severe_mucositis == a.y_true).all()
    assert a.model_center.isin(['B', 'C'] if cohort == 'Development' else ['A']).all()
    a['analysis_cohort'] = cohort
    blocks.append(a)
data = pd.concat(blocks, ignore_index=True)
assert set(data.patient_id) == set(m.patient_id)
data[['patient_id', 'model_center', 'analysis_cohort', 'RT_start_year', 'y_true', 'gtv_volume_cm3', 'M3']].to_csv(OUT / 'analysis_patients.csv', index=False, encoding='utf-8-sig')

def metrics(y, p):
    order = np.argsort(-p, kind='stable')
    ys, ps = (y[order], p[order])
    ends = np.r_[np.flatnonzero(np.diff(ps)), len(y) - 1]
    tp = np.cumsum(ys)[ends].astype(float)
    fp = ends + 1 - tp
    return np.array([np.trapezoid(np.r_[0, tp / y.sum()], np.r_[0, fp / (len(y) - y.sum())]), np.trapezoid(np.r_[1, tp / (tp + fp)], np.r_[0, tp / y.sum()]), np.mean((y - p) ** 2)])

def auc_rank(y, p):
    ranks = rankdata(p)
    n1 = y.sum()
    n0 = len(y) - n1
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.959963984540054
    den = 1 + z * z / n
    mid = (k / n + z * z / (2 * n)) / den
    half = z * np.sqrt(k / n * (1 - k / n) / n + z * z / (4 * n * n)) / den
    return [float(mid - half), float(mid + half)]
center = []
corr = []
years = []
for seed, (group, df) in enumerate([('Development', blocks[0]), ('External', blocks[1])] + [(c, data[data.model_center == c]) for c in ['A', 'B', 'C']], start=20261025):
    if group == 'A':
        center.append(dict(next((r for r in center if r['group'] == 'External')), group='A'))
        continue
    y = df.y_true.to_numpy()
    p = df.M3.to_numpy()
    v = df.gtv_volume_cm3.to_numpy()
    est = metrics(y, p)[0]
    assert abs(est - auc_rank(y, p)) < 1e-12
    rng = np.random.default_rng(seed)
    bc = []
    br = []
    for _ in range(5000):
        idx = rng.integers(0, len(df), len(df))
        if np.unique(y[idx]).size == 2:
            bc.append(auc_rank(y[idx], p[idx]))
        br.append(float(spearmanr(v[idx], p[idx]).statistic))
    cl, ch = np.quantile(bc, [0.025, 0.975])
    rl, rh = np.quantile(br, [0.025, 0.975])
    lo, hi = wilson(int(y.sum()), len(df))
    center.append(dict(group=group, n=len(df), events=int(y.sum()), rate=float(y.mean()), rate_ci_low=lo, rate_ci_high=hi, auc=float(est), auc_ci_low=float(cl), auc_ci_high=float(ch), bootstrap_valid=len(bc), seed=seed))
    if group != 'A':
        q = np.quantile(v, [0.25, 0.5, 0.75])
        corr.append(dict(group=group, n=len(df), volume_median=float(q[1]), volume_q1=float(q[0]), volume_q3=float(q[2]), rho=float(spearmanr(v, p).statistic), ci_low=float(rl), ci_high=float(rh), seed=seed))
for yr in sorted(data.RT_start_year.unique()):
    for group in ['All', 'A', 'B', 'C']:
        df = data[(data.RT_start_year == yr) & (True if group == 'All' else data.model_center == group)]
        n = len(df)
        k = int(df.y_true.sum())
        lo, hi = wilson(k, n)
        years.append(dict(year=int(yr), group=group, n=n, events=k, rate=k / n if n else None, ci_low=lo, ci_high=hi))
print('Centers:', json.dumps(center, ensure_ascii=False), flush=True)
print('Correlations:', json.dumps(corr, ensure_ascii=False), flush=True)
models = ['M2_Abl', 'M2', 'M3_Abl', 'M3', 'M4']
contrasts = [('M2', 'M2_Abl', 'Oral-conditioned dose'), ('M3_Abl', 'M2_Abl', 'GTV mask'), ('M3', 'M3_Abl', 'Joint structure-conditioned dose'), ('M4', 'M3', 'Planning CT')]
qc = pd.read_excel(qcfile, sheet_name='QC_paired_comparisons') if qcfile else pd.DataFrame(columns=['Cohort', 'Candidate', 'Reference', 'Metric', 'P value', 'Improvement'])
print('QC columns:', qc.columns.tolist(), flush=True)
tests = []
B = 20000
for ci, df in enumerate(blocks):
    cohort = df.analysis_cohort.iloc[0]
    y = df.y_true.to_numpy()
    pp = df[models].to_numpy()
    seed = 20261025 + ci
    rng = np.random.default_rng(seed)
    strata = df.model_center.astype(str) + '_' + df.y_true.astype(str)
    indexes = [np.flatnonzero(strata.to_numpy() == s) for s in sorted(strata.unique())]
    observed = np.array([metrics(y, pp[:, i]) for i in range(len(models))])
    boot = np.empty((B, len(models), 3))
    for b in range(B):
        ix = np.concatenate([rng.choice(ind, size=len(ind), replace=True) for ind in indexes])
        for mi in range(len(models)):
            boot[b, mi] = metrics(y[ix], pp[ix, mi])
        if (b + 1) % 5000 == 0:
            print(cohort, b + 1, flush=True)
    for candidate, reference, meaning in contrasts:
        ia, ib = (models.index(candidate), models.index(reference))
        for j, metric in enumerate(['AUROC', 'PR-AUC', 'Brier reduction']):
            sign = -1 if j == 2 else 1
            est = float((observed[ia, j] - observed[ib, j]) * sign)
            vals = (boot[:, ia, j] - boot[:, ib, j]) * sign
            lo, hi = np.quantile(vals, [0.025, 0.975])
            pv = min(1.0, 2 * (min(int((vals <= 0).sum()), int((vals >= 0).sum())) + 1) / (B + 1))
            candidates = qc[qc['Cohort'].astype(str).str.contains(cohort, case=False) & (qc['Candidate'].astype(str).str.replace('-', '_') == candidate) & (qc['Reference'].astype(str).str.replace('-', '_') == reference)]
            wanted = 'Brier' if j == 2 else 'PR' if j == 1 else 'AUROC'
            candidates = candidates[candidates.Metric.astype(str).str.contains(wanted, case=False)]
            if j == 0:
                candidates = candidates[~candidates.Metric.astype(str).str.contains('PR', case=False)]
            orig = None if len(candidates) == 0 else float(candidates.iloc[0]['P value'])
            if len(candidates):
                assert abs(float(candidates.iloc[0]['Improvement']) - est) < 1e-05
            tests.append(dict(cohort=cohort, candidate=candidate, reference=reference, meaning=meaning, metric=metric, improvement=est, ci_low=float(lo), ci_high=float(hi), p_bootstrap=pv, original_p=orig, seed=seed))

def holm(p):
    p = np.array(p)
    ix = np.argsort(p, kind='stable')
    result = np.empty(len(p))
    result[ix] = np.minimum(1, np.maximum.accumulate(p[ix] * np.arange(len(p), 0, -1)))
    assert (result >= p - 1e-12).all()
    return result
globalp = holm([r['p_bootstrap'] for r in tests])
for r, v in zip(tests, globalp):
    r['p_holm_24'] = float(v)
for cohort in ['Development', 'External']:
    rr = [r for r in tests if r['cohort'] == cohort]
    for r, v in zip(rr, holm([r['p_bootstrap'] for r in rr])):
        r['p_holm_12'] = float(v)
assert len(tests) == 24
sources = [master, snapshot, source / 'Development_aligned_predictions.csv', source / 'External_aligned_predictions.csv']
audit = [dict(path=str(x.resolve()), sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in sources]
result = dict(center=center, years=years, correlations=corr, tests=tests, source_audit=audit, bootstrap_comparisons=B, bootstrap_center_correlation=5000, master_snapshot_verified=True, created=datetime.now().isoformat())
(OUT / 'additional_analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
for key in ['center', 'years', 'correlations', 'tests']:
    pd.DataFrame(result[key]).to_csv(OUT / f'{key}.csv', index=False, encoding='utf-8-sig')
print(pd.DataFrame(tests)[['cohort', 'candidate', 'reference', 'metric', 'p_bootstrap', 'p_holm_24', 'p_holm_12']].to_string(index=False), flush=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), sharey=True)
colors = {'A': '#0072B2', 'B': '#D55E00', 'C': '#009E73'}
for ax, df, label in zip(axes, blocks, ['Development (OOF)', 'Locked external validation']):
    for c, g in df.groupby('model_center'):
        ax.scatter(g.gtv_volume_cm3, g.M3, s=19, alpha=0.68, color=colors[c], label=f'Center {c} (n = {len(g)})', linewidths=0)
    cr = next((r for r in corr if r['group'] == df.analysis_cohort.iloc[0]))
    ax.set_xscale('log')
    ax.set_xlim(data.gtv_volume_cm3.min() * 0.8, data.gtv_volume_cm3.max() * 1.2)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('Primary-GTV volume (cm³; logarithmic scale)')
    ax.set_title(label, fontsize=11, fontweight='bold', pad=9)
    ax.text(0.04, 0.96, f"Spearman ρ = {cr['rho']:.3f}\n95% CI {cr['ci_low']:.3f} to {cr['ci_high']:.3f}", transform=ax.transAxes, va='top', fontsize=9, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9})
    ax.legend(loc='lower right', frameon=False, fontsize=9)
    ax.grid(axis='y', color='#dddddd', linewidth=0.5)
    ax.set_axisbelow(True)
axes[0].set_ylabel('Frozen M3 predicted probability')
fig.tight_layout(pad=1.2)
fig.savefig(OUT / 'Figure_S1_M3_risk_GTV_volume.png', dpi=300)
fig.savefig(OUT / 'Figure_S1_M3_risk_GTV_volume.pdf')
plt.close(fig)
print('DONE', OUT, flush=True)
register('additional_analysis_dir', str(OUT))
