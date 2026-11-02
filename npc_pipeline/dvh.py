from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
from .common import config, load_cohort, require, new_output, register, sha256, write_json, integer_values
from .preprocess import resolve_image as resolve
THRESHOLDS=list(range(5,71,5))
def same_grid(a, b):
    return a.GetSize() == b.GetSize() and all((np.allclose(x, y, atol=1e-05, rtol=0) for x, y in [(a.GetSpacing(), b.GetSpacing()), (a.GetOrigin(), b.GetOrigin()), (a.GetDirection(), b.GetDirection())]))

def process(row):
    pid = int(row['patient_id'])
    base = {'patient_id': pid, 'model_center': row['model_center'], 'included_final489': 1}
    qc = {'patient_id': pid, 'model_center': row['model_center']}
    try:
        paths = {k: resolve(row[k], pid) for k in ['oral_cavity_mask_path', 'gtv_mask_path', 'dose_path']}
        before = {k: (p.stat().st_size, p.stat().st_mtime_ns) for k, p in paths.items()}
        oral = sitk.ReadImage(str(paths['oral_cavity_mask_path']))
        gtv = sitk.ReadImage(str(paths['gtv_mask_path']))
        dose = sitk.ReadImage(str(paths['dose_path']))
        scale=float(row['dose_scale_to_gy'])
        if not np.isfinite(scale) or not 0 < scale <= 10: raise ValueError('Explicit dose_scale_to_gy required')
        if scale != 1.0: dose = sitk.Cast(dose, sitk.sitkFloat64) * scale
        for name, im in [('oral', oral), ('gtv', gtv), ('dose', dose)]:
            if im.GetDimension() != 3 or im.GetNumberOfComponentsPerPixel() != 1:
                raise ValueError(f'{name}: expected scalar 3D image')
            if not all(np.isfinite(im.GetSpacing())) or min(im.GetSpacing()) <= 0:
                raise ValueError(f'{name}: invalid spacing')
        oa = sitk.GetArrayViewFromImage(oral)
        ga = sitk.GetArrayViewFromImage(gtv)
        if not np.isfinite(oa).all() or not np.isfinite(ga).all():
            raise ValueError('Nonfinite mask')
        ob = oa > 0.5
        gb = ga > 0.5
        on = int(ob.sum())
        gn = int(gb.sum())
        if min(on, gn) == 0:
            raise ValueError('Empty oral/GTV mask')
        ov = float(np.prod(oral.GetSpacing()) / 1000)
        gv = float(np.prod(gtv.GetSpacing()) / 1000)
        oral_volume = on * ov
        gtv_volume = gn * gv
        gm = not same_grid(gtv, oral)
        dm = not same_grid(dose, oral)
        if gm:
            gtv_on_oral = sitk.Resample(gtv, oral, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            overlap_n = int(np.count_nonzero(ob & (sitk.GetArrayViewFromImage(gtv_on_oral) > 0.5)))
        else:
            overlap_n = int(np.count_nonzero(ob & gb))
        da = sitk.GetArrayViewFromImage(dose)
        if not np.isfinite(da).all():
            raise ValueError('Nonfinite dose')
        dmin = float(da.min())
        dmax = float(da.max())
        if dmin < -1e-05 or dmax <= 0 or dmax > 150:
            raise ValueError(f'Dose scale requires review: {dmin}, {dmax}')
        coverage = 100.0
        if dm:
            cover = sitk.Image(dose.GetSize(), sitk.sitkUInt8)
            cover.CopyInformation(dose)
            cover = cover + 1
            cover = sitk.Resample(cover, oral, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
            coverage = float(np.count_nonzero(sitk.GetArrayViewFromImage(cover)[ob] > 0) / on * 100)
            if coverage < 99.999999:
                raise ValueError(f'Incomplete oral dose coverage: {coverage}%')
            dose_on_oral = sitk.Resample(dose, oral, sitk.Transform(3, sitk.sitkIdentity), sitk.sitkLinear, 0, sitk.sitkFloat32)
            values = np.asarray(sitk.GetArrayViewFromImage(dose_on_oral)[ob], dtype=np.float64)
        else:
            values = np.asarray(da[ob], dtype=np.float64)
        result = {**base, 'gtv_volume_cm3': gtv_volume, 'oral_volume_cm3': oral_volume, 'gtv_oral_overlap_volume_cm3': overlap_n * ov, 'oral_Dmean_Gy': float(values.mean()), 'oral_Dmax_Gy': float(values.max()), 'oral_D2_Gy': float(np.percentile(values, 98, method='linear'))}
        for t in THRESHOLDS:
            result[f'oral_V{t}_percent'] = float(np.count_nonzero(values >= t) / on * 100)
        vx = [result[f'oral_V{t}_percent'] for t in THRESHOLDS]
        assert all((0 <= v <= 100 for v in vx)) and all((a >= b for a, b in zip(vx, vx[1:])))
        assert result['oral_Dmean_Gy'] <= result['oral_Dmax_Gy'] + 1e-08
        assert result['oral_D2_Gy'] <= result['oral_Dmax_Gy'] + 1e-08
        assert before == {k: (p.stat().st_size, p.stat().st_mtime_ns) for k, p in paths.items()}, 'Source changed during reading'
        binary = bool(np.all((oa == 0) | (oa == 1)) and np.all((ga == 0) | (ga == 1)))
        result['extraction_status'] = 'PASS' if binary else 'REVIEW_NONBINARY_MASK'
        qc.update({'extraction_status': result['extraction_status'], 'oral_mask_voxels': on, 'gtv_mask_voxels': gn, 'oral_voxel_volume_mm3': ov * 1000, 'gtv_voxel_volume_mm3': gv * 1000, 'oral_dose_coverage_percent': coverage, 'gtv_resampled_to_oral': int(gm), 'dose_resampled_to_oral': int(dm), 'dose_scale_to_gy': scale, 'source_dose_min': dmin, 'source_dose_max': dmax, 'source_dose_unit': 'Gy; explicit per-case dose_scale_to_gy; not a new DICOM audit', 'oral_source_path': str(paths['oral_cavity_mask_path']), 'gtv_source_path': str(paths['gtv_mask_path']), 'dose_source_path': str(paths['dose_path']), 'error': ''})
        return (result, qc)
    except Exception as e:
        return ({**base, 'extraction_status': 'ERROR'}, {**qc, 'extraction_status': 'ERROR', 'error': str(e)})

def run():
    c=config(); raw,included=load_cohort(c)
    scales=pd.read_csv(c['dose_scale_csv'])
    scales['patient_id']=integer_values(scales.patient_id,'dose QC patient_id')
    require(not scales.patient_id.duplicated().any(),'Duplicate dose QC IDs')
    require(set(included.patient_id).issubset(scales.patient_id),'Missing dose QC IDs')
    rows=included.merge(scales[['patient_id','dose_scale_to_gy']],on='patient_id',validate='one_to_one').to_dict('records')
    out=new_output('GTV_DVH_final489'); results=[]; audits=[]
    for i,row in enumerate(rows,1):
        result,audit=process(row); results.append(result); audits.append(audit)
        if i%25==0 or i==len(rows):print(f'Extracted {i}/{len(rows)}',flush=True)
    data=pd.DataFrame(results); qc=pd.DataFrame(audits)
    # Output in the original master order; excluded records contain no extracted metrics.
    base=raw[['patient_id','model_center','exclude_reason']].copy()
    aligned=base.merge(data.drop(columns=['model_center']),on='patient_id',how='left',sort=False,validate='one_to_one')
    aligned.loc[aligned.exclude_reason.fillna('').astype(str).str.strip().ne(''),'extraction_status']='EXCLUDED_NOT_EXTRACTED'
    require(aligned.patient_id.tolist()==raw.patient_id.tolist(),'Master ordering changed')
    with pd.ExcelWriter(out/'GTV_DVH_final489.xlsx',engine='openpyxl') as w:
        aligned.to_excel(w,sheet_name='Master_order',index=False)
        data.to_excel(w,sheet_name='Included_489',index=False)
        qc.to_excel(w,sheet_name='QC',index=False)
    data.to_csv(out/'metrics_489.csv',index=False,encoding='utf-8-sig')
    qc.to_csv(out/'QC.csv',index=False,encoding='utf-8-sig')
    write_json(out/'status.json',{'status_counts':data.extraction_status.value_counts().to_dict(),
        'master_sha256':sha256(c['master']),'dose_scale_sha256':sha256(c['dose_scale_csv'])})
    require(data.extraction_status.eq('PASS').all(),f'Extraction needs review: {out}')
    register('dvh_dir',str(out))
    return out
