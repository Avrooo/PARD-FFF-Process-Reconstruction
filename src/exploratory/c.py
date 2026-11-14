import pandas as pd, numpy as np, warnings, glob; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.metrics import roc_auc_score
d=pd.read_pickle('d.pkl')
ra=pd.read_excel(glob.glob('/mnt/user-data/uploads/*.xlsx')[0],sheet_name='Run_Audit')[['run_id','fault_type','schedule_family','max_severity']]
d=d.merge(ra,on='run_id')
print(ra.groupby(['fault_type','schedule_family']).size().unstack(fill_value=0))
lab=['fault_code','any_abnormal','severity_code','thermal_flag','cooling_flag','speed_flag','underextrusion_flag','fault_type','schedule_family','max_severity']
cmd=[c for c in d.columns if 'cmd' in c or c in ('layer_z','planned_velocity_mean_mm_s','speed_factor_mean','extrude_factor_mean')]
sens=[c for c in d.columns if c not in lab+cmd+['run_id','layer_id','commanded_extrusion_positive_delta','extrusion_per_path_length','toolpath_length_xy']]
# run-relative normalisation: subtract run's median of layers 2..10 (assumed early nominal)
ref=d[(d.layer_id>=2)&(d.layer_id<=10)].groupby('run_id')[sens].median()
rel=d[sens].values-ref.loc[d.run_id].values
R=pd.DataFrame(rel,columns=[s+'_rel' for s in sens],index=d.index)
tr=d.fault_type!='Mixed: Thermal+Cooling+Speed+Flow'; te=~tr
print('train single',tr.sum(),'test mixed',te.sum())
for t in ['thermal_flag','cooling_flag','speed_flag','underextrusion_flag']:
    out=[]
    for name,X in [('raw',d[sens]),('run-relative',R),('both',pd.concat([d[sens],R],axis=1))]:
        m=H(max_iter=150).fit(X[tr],d.loc[tr,t]); out.append(roc_auc_score(d.loc[te,t],m.predict_proba(X[te])[:,1]))
    print(t,'single->mixed AUC raw %.3f rel %.3f both %.3f'%tuple(out))
# family shift: train gradual+step -> test mixed schedule
for fam in ['gradual','step','mixed']:
    tr2=d.schedule_family!=fam; te2=~tr2
    s=[]
    for t in ['thermal_flag','cooling_flag','speed_flag','underextrusion_flag']:
        m=H(max_iter=150).fit(d.loc[tr2,sens],d.loc[tr2,t]); s.append(round(roc_auc_score(d.loc[te2,t],m.predict_proba(d.loc[te2,sens])[:,1]),3))
    print('leave family out',fam,s)
