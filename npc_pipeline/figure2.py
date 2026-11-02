from .common import config, new_output, load_cohort, register
from .predictions import load_aligned
c = config()
load_cohort(c)
load_aligned()
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
PROJECT_ROOT = Path(c['result_root'])
MASTER_PATH = Path(c['master'])
DEV_DIR = Path(c['prediction_dir']) / 'development'
EXT_DIR = Path(c['prediction_dir']) / 'external'
OUT_DIR = new_output('Figure2')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / 'Figure2_RnO.png'
OUT_PDF = OUT_DIR / 'Figure2_RnO.pdf'
OUT_SVG = OUT_DIR / 'Figure2_RnO.svg'
OUT_XLSX = OUT_DIR / 'Figure2_RnO_source_data.xlsx'
OUT_JSON = OUT_DIR / 'Figure2_RnO_audit.json'
DEV_FILES = {'M1': 'M1_Dose_Only_patient_averaged_oof_309.csv', 'M2-Abl': 'M2_Abl_patient_averaged_oof_309.csv', 'M2': 'M2_Oral_MaskedDose_patient_averaged_oof_309.csv', 'M3-Abl': 'M3_Abl_patient_averaged_oof_309.csv', 'M3': 'M3_Oral_GTV_MaskedDose_patient_averaged_oof_309.csv', 'M4': 'M4_CT_Augmented_M3_patient_averaged_oof_309.csv'}
PLOT_ORDER = ['M1', 'M2-Abl', 'M2', 'M3-Abl', 'M4', 'M3']
LEGEND_ORDER = ['M1', 'M2-Abl', 'M2', 'M3-Abl', 'M3', 'M4']
LINESTYLES = {'M1': '-', 'M2-Abl': '--', 'M2': '-.', 'M3-Abl': '-', 'M3': '-', 'M4': '--'}
LINEWIDTHS = {'M1': 1.5, 'M2-Abl': 1.5, 'M2': 1.5, 'M3-Abl': 1.7, 'M4': 1.6, 'M3': 2.8}

def assert_unique_ids(frame: pd.DataFrame, name: str) -> None:
    if frame['patient_id'].duplicated().any():
        raise RuntimeError(f'{name} contains duplicate patient_id values')
master = pd.read_excel(MASTER_PATH, sheet_name=c.get('master_sheet', 0))
required_master_columns = {'patient_id', 'model_center', 'severe_mucositis', 'exclude_reason'}
missing = required_master_columns - set(master.columns)
if missing:
    raise RuntimeError(f'Master is missing columns: {sorted(missing)}')
master['patient_id'] = pd.to_numeric(master['patient_id'], errors='raise').astype(int)
master['model_center'] = master['model_center'].astype(str).str.strip().str.upper()
master['severe_mucositis'] = pd.to_numeric(master['severe_mucositis'], errors='coerce')
exclude_text = master['exclude_reason'].fillna('').astype(str).str.strip()
master_analysis = master.loc[exclude_text.eq('') & master['model_center'].isin(['A', 'B', 'C']) & master['severe_mucositis'].isin([0, 1]), ['patient_id', 'severe_mucositis', 'model_center']].copy()
master_analysis['severe_mucositis'] = master_analysis['severe_mucositis'].astype(int)
assert_unique_ids(master_analysis, 'Final analysis master')
master_dev = master_analysis.loc[master_analysis['model_center'].isin(['B', 'C'])].sort_values('patient_id').reset_index(drop=True)
master_ext = master_analysis.loc[master_analysis['model_center'].eq('A')].sort_values('patient_id').reset_index(drop=True)
if (len(master_analysis), len(master_dev), len(master_ext)) != (489, 309, 180):
    raise RuntimeError(f'Final cohort mismatch: expected total/development/external = 489/309/180, observed {len(master_analysis)}/{len(master_dev)}/{len(master_ext)}')
dev_wide = None
for display_name, filename in DEV_FILES.items():
    frame = pd.read_csv(DEV_DIR / filename)
    frame['patient_id'] = pd.to_numeric(frame['patient_id'], errors='raise').astype(int)
    frame['y_true'] = pd.to_numeric(frame['y_true'], errors='raise').astype(int)
    frame['oof_probability_mean'] = pd.to_numeric(frame['oof_probability_mean'], errors='raise')
    assert_unique_ids(frame, f'Development predictions for {display_name}')
    subset = frame[['patient_id', 'y_true', 'oof_probability_mean']].rename(columns={'oof_probability_mean': display_name})
    dev_wide = subset if dev_wide is None else dev_wide.merge(subset, on=['patient_id', 'y_true'], how='inner', validate='one_to_one')
dev_wide = dev_wide.sort_values('patient_id').reset_index(drop=True)
ext_long = pd.read_csv(EXT_DIR / 'external_patient_20model_ensemble_all_models.csv')
ext_long['patient_id'] = pd.to_numeric(ext_long['patient_id'], errors='raise').astype(int)
ext_long['y_true'] = pd.to_numeric(ext_long['y_true'], errors='raise').astype(int)
ext_long['probability'] = pd.to_numeric(ext_long['probability'], errors='raise')
if ext_long.duplicated(['display_name', 'patient_id']).any():
    raise RuntimeError('External predictions contain duplicate model/patient rows')
ext_wide = ext_long[['display_name', 'patient_id', 'y_true', 'probability']].pivot(index=['patient_id', 'y_true'], columns='display_name', values='probability').reset_index().sort_values('patient_id').reset_index(drop=True)
if len(dev_wide) != 309 or set(dev_wide['patient_id']) != set(master_dev['patient_id']):
    raise RuntimeError('Development predictions do not match the locked 309-patient cohort')
if len(ext_wide) != 180 or set(ext_wide['patient_id']) != set(master_ext['patient_id']):
    raise RuntimeError('External predictions do not match the locked 180-patient cohort')

def compute_curves(frame: pd.DataFrame, column: str) -> dict:
    y_true = frame['y_true'].to_numpy()
    probability = frame[column].to_numpy()
    fpr, tpr, _ = roc_curve(y_true, probability)
    precision, recall, _ = precision_recall_curve(y_true, probability)
    return {'fpr': fpr, 'tpr': tpr, 'precision': precision, 'recall': recall, 'roc_auc_curve': float(auc(fpr, tpr)), 'pr_auc_curve': float(auc(recall, precision))}
dev_curves = {model: compute_curves(dev_wide, model) for model in LEGEND_ORDER}
ext_curves = {model: compute_curves(ext_wide, model) for model in LEGEND_ORDER}
dev_prev = float(dev_wide['y_true'].mean())
ext_prev = float(ext_wide['y_true'].mean())
metric_fingerprint = pd.DataFrame([{'display_name': model, 'roc_auc_dev': dev_curves[model]['roc_auc_curve'], 'pr_auc_dev': dev_curves[model]['pr_auc_curve'], 'roc_auc_ext': ext_curves[model]['roc_auc_curve'], 'pr_auc_ext': ext_curves[model]['pr_auc_curve']} for model in LEGEND_ORDER]).sort_values('display_name').reset_index(drop=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8.5})
fig = plt.figure(figsize=(11.6, 8.6), facecolor='white')
grid = fig.add_gridspec(2, 2, left=0.08, right=0.985, top=0.9, bottom=0.16, wspace=0.18, hspace=0.23)
ax_a, ax_b, ax_c, ax_d = [fig.add_subplot(grid[row, col]) for row, col in [(0, 0), (0, 1), (1, 0), (1, 1)]]
legend_handles = {}
for model in PLOT_ORDER:
    line, = ax_a.plot(dev_curves[model]['fpr'], dev_curves[model]['tpr'], linestyle=LINESTYLES[model], linewidth=LINEWIDTHS[model], label=model, zorder=5 if model == 'M3' else 3)
    legend_handles[model] = line
ax_a.plot([0, 1], [0, 1], linestyle=':', linewidth=1.0, color='0.55', zorder=1)
ax_a.set(xlim=(0, 1), ylim=(0, 1), xlabel='1 − Specificity', ylabel='Sensitivity', title='ROC')
ax_a.grid(True, linewidth=0.4, alpha=0.35)
for model in PLOT_ORDER:
    ax_b.plot(ext_curves[model]['fpr'], ext_curves[model]['tpr'], linestyle=LINESTYLES[model], linewidth=LINEWIDTHS[model], label=model, zorder=5 if model == 'M3' else 3)
ax_b.plot([0, 1], [0, 1], linestyle=':', linewidth=1.0, color='0.55', zorder=1)
ax_b.set(xlim=(0, 1), ylim=(0, 1), xlabel='1 − Specificity', ylabel='Sensitivity', title='ROC')
ax_b.grid(True, linewidth=0.4, alpha=0.35)
for model in PLOT_ORDER:
    ax_c.plot(dev_curves[model]['recall'], dev_curves[model]['precision'], linestyle=LINESTYLES[model], linewidth=LINEWIDTHS[model], label=model, zorder=5 if model == 'M3' else 3)
ax_c.axhline(dev_prev, linestyle=':', linewidth=1.0, color='0.55', zorder=1)
ax_c.text(0.985, min(dev_prev + 0.012, 0.98), f'Prevalence = {dev_prev:.3f}', ha='right', va='bottom', fontsize=8, color='0.35')
ax_c.set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall', ylabel='Precision', title='Precision–recall')
ax_c.grid(True, linewidth=0.4, alpha=0.35)
for model in PLOT_ORDER:
    ax_d.plot(ext_curves[model]['recall'], ext_curves[model]['precision'], linestyle=LINESTYLES[model], linewidth=LINEWIDTHS[model], label=model, zorder=5 if model == 'M3' else 3)
ax_d.axhline(ext_prev, linestyle=':', linewidth=1.0, color='0.55', zorder=1)
ax_d.text(0.985, min(ext_prev + 0.012, 0.98), f'Prevalence = {ext_prev:.3f}', ha='right', va='bottom', fontsize=8, color='0.35')
ax_d.set(xlim=(0, 1), ylim=(0, 1), xlabel='Recall', ylabel='Precision', title='Precision–recall')
ax_d.grid(True, linewidth=0.4, alpha=0.35)
for axis, label in zip([ax_a, ax_b, ax_c, ax_d], ['A', 'B', 'C', 'D']):
    axis.text(-0.16, 1.05, label, transform=axis.transAxes, fontsize=14, fontweight='bold', ha='left', va='bottom')
fig.text(0.285, 0.94, 'Development cohort', ha='center', va='center', fontsize=11, fontweight='bold')
fig.text(0.755, 0.94, 'Independent external validation cohort', ha='center', va='center', fontsize=11, fontweight='bold')
legend = fig.legend([legend_handles[model] for model in LEGEND_ORDER], LEGEND_ORDER, loc='lower center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.065), handlelength=2.8, columnspacing=1.6)
for label in legend.get_texts():
    if label.get_text() == 'M3':
        label.set_fontweight('bold')
fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_PDF, bbox_inches='tight', facecolor='white')
fig.savefig(OUT_SVG, bbox_inches='tight', facecolor='white')
plt.show()
with pd.ExcelWriter(OUT_XLSX, engine='openpyxl') as writer:
    master_dev.to_excel(writer, sheet_name='master_dev_309', index=False)
    master_ext.to_excel(writer, sheet_name='master_ext_180', index=False)
    dev_wide.to_excel(writer, sheet_name='dev_predictions_wide', index=False)
    ext_wide.to_excel(writer, sheet_name='ext_predictions_wide', index=False)
    metric_fingerprint.to_excel(writer, sheet_name='metric_fingerprint', index=False)
    for model in LEGEND_ORDER:
        pd.DataFrame({'fpr': dev_curves[model]['fpr'], 'tpr': dev_curves[model]['tpr']}).to_excel(writer, sheet_name=f'ROC_dev_{model}', index=False)
        pd.DataFrame({'fpr': ext_curves[model]['fpr'], 'tpr': ext_curves[model]['tpr']}).to_excel(writer, sheet_name=f'ROC_ext_{model}', index=False)
        pd.DataFrame({'recall': dev_curves[model]['recall'], 'precision': dev_curves[model]['precision']}).to_excel(writer, sheet_name=f'PR_dev_{model}', index=False)
        pd.DataFrame({'recall': ext_curves[model]['recall'], 'precision': ext_curves[model]['precision']}).to_excel(writer, sheet_name=f'PR_ext_{model}', index=False)
audit = {'generated_utc': datetime.now(timezone.utc).isoformat(), 'final_analysis_total': len(master_analysis), 'development_n': len(master_dev), 'external_n': len(master_ext), 'development_definition': 'model_center B+C', 'external_definition': 'model_center A', 'exclusion_rule': 'exclude_reason must be blank', 'pr_auc_definition': 'trapezoidal integration of precision versus recall, with recall on the x-axis', 'metric_fingerprint': metric_fingerprint.to_dict(orient='records')}
OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
print('=' * 92)
print(f'Output folder: {OUT_DIR}')
print('Generated Figure 2 PNG/PDF/SVG, source-data XLSX, and audit JSON.')
register('figure2_dir', str(OUT_DIR))
