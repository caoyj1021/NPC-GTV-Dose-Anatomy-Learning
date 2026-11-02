"""Synthetic regression tests; no governed data required."""
from pathlib import Path
import ast
import json
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score, log_loss
from npc_pipeline.common import COUNTS, EXCLUSIONS, ROLES, validate_cohort, load_cohort, config, checked_predictions
from npc_pipeline import splits

def synthetic_master():
    rows=[]
    for (center,y),n in COUNTS.items():
        for _ in range(n):
            rows.append(dict(patient_id=10001+len(rows),model_center=center,cohort=ROLES[center],severe_mucositis=y,exclude_reason=''))
    for reason,n in EXCLUSIONS.items():
        for _ in range(n):
            rows.append(dict(patient_id=10001+len(rows),model_center='A',cohort='External',severe_mucositis=0,exclude_reason=reason))
    return pd.DataFrame(rows)

@pytest.fixture
def configured(tmp_path,monkeypatch):
    master=tmp_path/'master.xlsx';synthetic_master().to_excel(master,index=False)
    c={'master':str(master),'master_sheet':0,'result_root':str(tmp_path/'results'),'npz_dir':str(tmp_path/'npz')}
    path=tmp_path/'config.local.json';path.write_text(json.dumps(c),encoding='utf-8')
    monkeypatch.setenv('NPC_CONFIG',str(path))
    return c

def test_final_cohort_and_center_roles():
    _,d=validate_cohort(synthetic_master())
    assert len(d)==489 and (d.cohort=='Development').sum()==309
    assert d.loc[d.model_center=='A','cohort'].eq('External').all()

@pytest.mark.parametrize('issue',['wrong_role','duplicate','changed_label','wrong_exclusions','noninteger_id','missing_label'])
def test_cohort_rejects_inconsistent_inputs(issue):
    raw=synthetic_master()
    if issue=='wrong_role':raw.loc[raw.model_center=='A','cohort']='Development'
    elif issue=='duplicate':raw.loc[1,'patient_id']=raw.loc[0,'patient_id']
    elif issue=='changed_label':raw.loc[0,'severe_mucositis']=1
    elif issue=='wrong_exclusions':raw.loc[539,'exclude_reason']=''
    elif issue=='noninteger_id':raw['patient_id']=raw.patient_id.astype(float);raw.loc[0,'patient_id']=1.5
    elif issue=='missing_label':raw.loc[0,'severe_mucositis']=np.nan
    with pytest.raises(ValueError):validate_cohort(raw)

def test_splits_cover_each_development_patient_once_and_no_external_leakage(configured):
    out=splits.run();frames,configs,_=splits.load_locked()
    _,included=load_cohort();external=set(included[included.model_center=='A'].patient_id)
    for r in range(1,5):
        validation=[]
        for f in range(1,6):
            parts=frames[r,f]
            assert configs[r,f]['training_seed']==30260700+r*10+f
            validation.extend(parts['outer_validation'].patient_id)
            for part in parts.values():assert not set(part.patient_id)&external
        assert len(validation)==len(set(validation))==309
    assert out.is_dir()

def test_splits_reject_changed_content_after_lock(configured):
    out=splits.run()
    path=out/'repeat_01/fold_01/outer_train.csv'
    path.write_text(path.read_text()+'\n',encoding='utf-8')
    with pytest.raises(ValueError,match='changed'):splits.load_locked()

def test_outputs_are_not_overwritten(configured):
    a=splits.run();b=splits.run()
    assert a!=b and a.is_dir() and b.is_dir()

def test_predictions_reject_wrong_identity_label_and_probability():
    _,d=validate_cohort(synthetic_master())
    ref=d[d.model_center=='A']
    z=ref[['patient_id','severe_mucositis']].rename(columns={'severe_mucositis':'y_true'})
    z['probability']=.5
    assert len(checked_predictions(z,ref))==180
    for col,value in [('patient_id',99999),('y_true',9),('probability',1.2)]:
        bad=z.copy();bad.loc[bad.index[0],col]=value
        with pytest.raises(ValueError):checked_predictions(bad,ref)

@pytest.mark.parametrize('p',[[0,.25,.25,1,.75,.75],[.5]*6,[0,1,0,1,1,0],[1,0,1,0,0,1]])
def test_metrics_match_independent_reference_with_ties(p):
    from npc_pipeline.paired import metrics
    y=np.array([0,1,0,1,1,0]);p=np.array(p,dtype=float)
    precision,recall,_=precision_recall_curve(y,p)
    np.testing.assert_allclose(metrics(y,p),[roc_auc_score(y,p),auc(recall,precision),np.mean((y-p)**2),log_loss(y,p)],atol=1e-12,rtol=0)

def test_pr_auc_is_trapezoidal_not_average_precision():
    from npc_pipeline.cnn_engine import safe_pr_auc
    y=np.array([0,1,0,1,1,0]);p=np.array([.05,.55,.45,.9,.35,.2])
    pr,rc,_=precision_recall_curve(y,p)
    assert np.isclose(safe_pr_auc(y,p),auc(rc,pr))
    assert not np.isclose(safe_pr_auc(y,p),average_precision_score(y,p))

def test_dca_counts_and_wilson_bins():
    from npc_pipeline.calibration import decision_curve_rows,equal_frequency_calibration_bins
    y=np.array([0,1,0,1]);p=np.array([.1,.8,.7,.3])
    d=decision_curve_rows(y,p,np.array([.5]))
    assert d.iloc[0]['tp']==1 and d.iloc[0]['fp']==1
    assert d.iloc[0]['net_benefit_model']==0
    bins=equal_frequency_calibration_bins(np.tile([0,1],50),np.linspace(.001,.999,100))
    assert len(bins)==10 and bins.n.sum()==100 and bins.n.eq(10).all()

def test_holm_adjustment_against_statsmodels():
    from statsmodels.stats.multitest import multipletests
    p=Path(__file__).resolve().parents[1]/'npc_pipeline/additional_analysis.py'
    node=next(n for n in ast.parse(p.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='holm')
    ns={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
    values=[.01,.1,.005,.04,1,.01,.001,.9]
    np.testing.assert_allclose(ns['holm'](values),multipletests(values,method='holm')[1],atol=1e-14)

def test_npz_metadata_cannot_override_master_and_bad_identity_rejected(tmp_path):
    from npc_pipeline.cnn_engine import NPCNPZDataset
    path=tmp_path/'123.npz';shape=(64,112,80)
    np.savez_compressed(path,patient_id=123,label=0,cohort='WRONG',oral=np.ones(shape,np.uint8),gtv=np.ones(shape,np.uint8))
    frame=pd.DataFrame([dict(patient_id=123,severe_mucositis=1,npz_path=str(path))])
    _,y,pid=NPCNPZDataset(frame,['oral','gtv'],False)[0]
    assert int(y)==1 and int(pid)==123
    frame.patient_id=456
    with pytest.raises(RuntimeError):NPCNPZDataset(frame,['oral','gtv'],False)[0]

def test_atomic_npz_saver_preserves_directory_and_refuses_overwrite(tmp_path):
    from npc_pipeline.preprocess import save_npz
    arrays={k:np.ones((2,2,2),np.float32) for k in ['ct','dose','oral','gtv']}
    path=tmp_path/'case.npz';save_npz(path,arrays,{'patient_id':1})
    assert tmp_path.is_dir() and path.is_file()
    with pytest.raises(ValueError):save_npz(path,arrays,{'patient_id':2})
    with np.load(path) as z:assert int(z['patient_id'])==1

def test_all_notebooks_are_clean():
    root=Path(__file__).resolve().parents[1]
    for p in root.rglob('*.ipynb'):
        for cell in json.loads(p.read_text(encoding='utf-8'))['cells']:
            if cell['cell_type']=='code':
                assert not cell['outputs'] and cell['execution_count'] is None
                compile(''.join(cell['source']),str(p),'exec')


@pytest.mark.parametrize('bad_id',[123.5,[123,456]])
def test_npz_identifier_is_one_integer(tmp_path,bad_id):
    from npc_pipeline.cnn_engine import NPCNPZDataset
    path=tmp_path/'123.npz';shape=(64,112,80)
    np.savez_compressed(path,patient_id=bad_id,oral=np.ones(shape,np.uint8),gtv=np.ones(shape,np.uint8))
    frame=pd.DataFrame([dict(patient_id=123,severe_mucositis=1,npz_path=str(path))])
    with pytest.raises(RuntimeError,match='finite integer'):
        NPCNPZDataset(frame,['oral','gtv'],False)[0]


@pytest.mark.parametrize('change',['output','input','prediction_context'])
def test_downstream_artifact_rejects_mismatched_files(configured,tmp_path,change):
    from npc_pipeline.common import register
    from npc_pipeline.artifacts import lock_artifact,verify_artifact
    splits.run()
    predictions=tmp_path/'predictions';predictions.mkdir()
    (predictions/'prediction_audit.json').write_text('{}',encoding='utf-8')
    register('prediction_dir',str(predictions))
    output=tmp_path/'calibration';output.mkdir()
    (output/'workbook.xlsx').write_bytes(b'synthetic workbook bytes')
    source=tmp_path/'source.txt';source.write_text('original input',encoding='utf-8')
    lock_artifact(output,'calibration',[source])
    verify_artifact(output,'calibration')
    if change=='output':(output/'workbook.xlsx').write_bytes(b'changed')
    elif change=='input':source.write_text('changed',encoding='utf-8')
    else:(predictions/'prediction_audit.json').write_text('{"changed":true}',encoding='utf-8')
    with pytest.raises(ValueError):verify_artifact(output,'calibration')


def test_clinical_export_checks_oof_folds_and_external_means():
    from npc_pipeline.predictions import check_clinical_ensemble
    _,d=validate_cohort(synthetic_master());frames,configs=splits.generated_frames(d)
    refs={'Development':d[d.cohort.eq('Development')],'External':d[d.cohort.eq('External')]}
    oofs=[];externals=[]
    for (r,f),parts in frames.items():
        a=parts['outer_validation'][['patient_id','model_center','severe_mucositis']].rename(columns={'severe_mucositis':'y_true'}).copy()
        a['repeat']=r;a['fold']=f;a['probability']=.4+r*.01;oofs.append(a)
        e=refs['External'][['patient_id']].copy();e['repeat']=r;e['fold']=f;e['probability']=.5+r*.01;externals.append(e)
    oof=pd.concat(oofs,ignore_index=True);ext=pd.concat(externals,ignore_index=True)
    means={}
    for name,long in [('Development',oof),('External',ext)]:
        z=refs[name][['patient_id','severe_mucositis']].rename(columns={'severe_mucositis':'y_true'}).copy()
        z['probability']=z.patient_id.map(long.groupby('patient_id').probability.mean())
        means[name]=checked_predictions(z,refs[name])
    check_clinical_ensemble(oof,ext,means['Development'],means['External'],refs,frames)
    bad=oof.copy();bad.loc[0,'fold']=2
    with pytest.raises(ValueError):check_clinical_ensemble(bad,ext,means['Development'],means['External'],refs,frames)
    bad=means['External'].copy();bad.loc[0,'probability']+=.1
    with pytest.raises(ValueError,match='mean'):check_clinical_ensemble(oof,ext,means['Development'],bad,refs,frames)


def test_gradcam_accepts_null_optional_checkpoint_root(tmp_path):
    source=Path(__file__).resolve().parents[1]/'npc_pipeline/gradcam.py'
    node=next(n for n in ast.parse(source.read_text(encoding='utf-8')).body
              if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='FORMAL_MODEL_ROOT' for x in n.targets))
    ns={'Path':Path,'c':{'result_root':str(tmp_path),'frozen_sources':{'cnn_root':None}}}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    assert ns['FORMAL_MODEL_ROOT']==tmp_path
