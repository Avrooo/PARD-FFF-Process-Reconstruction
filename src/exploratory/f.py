import pandas as pd, numpy as np, warnings, glob; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as HC, HistGradientBoostingRegressor as HR
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
d=pd.read_pickle('d.pkl'); p=pd.read_pickle('p.pkl')
ra=pd.read_excel(glob.glob('/mnt/user-data/uploads/*.xlsx')[0],sheet_name='Run_Audit')[['run_id','fault_type','schedule_family']]
geo=['base_path_length_xy_mm','base_extrusion_path_xy_mm','base_travel_path_xy_mm','move_count','direction_change_count','sharp_turns_per_100mm','perimeter_path_fraction','infill_path_fraction','skin_path_fraction','base_nominal_extrusion_mm','base_nominal_layer_duration_s']
d=d.merge(ra,on='run_id').merge(p[['run_id','layer_id']+geo],on=['run_id','layer_id'])
d['vol']=d.live_extrusion_positive_delta/d.layer_duration_s
d['pathspeed']=d.toolpath_length_xy/d.layer_duration_s
load=['fan_actual_mean','live_velocity_mean_mm_s','vol','layer_duration_s','pathspeed','bed_power_mean']
cond=load+geo+['layer_id']
targets=['ext_temp_mean_C','ext_temp_min_C','ext_temp_max_C','ext_temp_std_C','ext_power_mean','ext_power_max','heater_on_fraction','ext_temp_slope_C_per_s']
raw=[c for c in d.columns if c not in ['run_id','layer_id','fault_code','any_abnormal','severity_code','thermal_flag','cooling_flag','speed_flag','underextrusion_flag','fault_type','schedule_family','layer_z','planned_velocity_mean_mm_s','commanded_extrusion_positive_delta','extrusion_per_path_length','speed_factor_mean','extrude_factor_mean']+geo and 'cmd' not in c and c not in ('vol','pathspeed')]
mixed=d.fault_type.str.startswith('Mixed'); tr=~mixed; te=mixed
y=d.thermal_flag
print('test positives',y[te].mean().round(3),'train',y[tr].mean().round(3))
def ci(yv,s,runs,B=300,seed=0):
    rng=np.random.default_rng(seed); u=runs.unique(); out=[]
    for _ in range(B):
        pick=rng.choice(u,len(u)); idx=np.concatenate([np.where(runs.values==r)[0] for r in pick])
        if yv.values[idx].min()==yv.values[idx].max(): continue
        out.append(roc_auc_score(yv.values[idx],s[idx]))
    return np.percentile(out,[2.5,97.5])
def report(name,score):
    a=roc_auc_score(y[te],score); lo,hi=ci(y[te].reset_index(drop=True),score,d.run_id[te].reset_index(drop=True))
    print(f'{name:48s} AUC {a:.3f} [{lo:.3f},{hi:.3f}]  AP {average_precision_score(y[te],score):.3f}')
# baselines
report('B0 |ext_temp_mean-210| threshold',np.abs(d.loc[te,'ext_temp_mean_C']-210).values)
report('B1 |temp_error_mean| threshold',np.abs(d.loc[te,'temp_error_mean_C']).values)
m=HC(max_iter=150).fit(d.loc[tr,raw],y[tr]); report('B2 raw telemetry GBM',m.predict_proba(d.loc[te,raw])[:,1])
# nominal model: normal layers of training runs only
nm=tr&(d.any_abnormal==0)
R=pd.DataFrame(index=d.index)
for t in targets:
    r=HR(max_iter=200,learning_rate=0.05).fit(d.loc[nm,cond],d.loc[nm,t])
    # out-of-fold for train rows to avoid optimistic residuals
    R[t]=d[t]-r.predict(d[cond])
# OOF residuals for training rows by run-group folds
from sklearn.model_selection import GroupKFold
for t in targets:
    oof=np.zeros(len(d))
    runs=d.run_id[tr&nm]
    for a,b in GroupKFold(5).split(d.loc[nm],groups=d.run_id[nm]):
        r=HR(max_iter=200,learning_rate=0.05).fit(d.loc[nm].iloc[a][cond],d.loc[nm].iloc[a][t])
        oof[d.index.get_indexer(d.loc[nm].index[b])]=r.predict(d.loc[nm].iloc[b][cond])
    full=HR(max_iter=200,learning_rate=0.05).fit(d.loc[nm,cond],d.loc[nm,t]).predict(d[cond])
    pred=np.where(nm,oof,full)   # normal train layers OOF; abnormal train & all test: full model
    R[t]=d[t]-pred
Rn=R.add_suffix('_res')
report('F1 nominal residual |ext_temp_mean_res|',np.abs(Rn['ext_temp_mean_C_res'][te]).values)
lr=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)).fit(Rn[tr],y[tr]); report('F2 residuals + logistic',lr.predict_proba(Rn[te])[:,1])
g=HC(max_iter=150).fit(Rn[tr],y[tr]); report('F3 residuals + GBM',g.predict_proba(Rn[te])[:,1])
X=pd.concat([Rn,d[raw]],axis=1); g2=HC(max_iter=150).fit(X[tr],y[tr]); report('F4 residuals + raw GBM',g2.predict_proba(X[te])[:,1])
# ablation: nominal model without load covariates (geometry only) -> shows value of load compensation
cond2=geo+['layer_id']; R2=pd.DataFrame(index=d.index)
for t in targets:
    r=HR(max_iter=200,learning_rate=0.05).fit(d.loc[nm,cond2],d.loc[nm,t]); R2[t+'_res']=d[t]-r.predict(d[cond2])
lr2=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(R2[tr],y[tr]); report('A1 ablate load comp (geometry-only nominal)',lr2.predict_proba(R2[te])[:,1])
# in-distribution reference: grouped CV on all runs
print('sanity ext_temp_mean_C vs temp_cmd corr in mixed:',d.loc[te,['ext_temp_mean_C','temp_cmd_C']].corr().iloc[0,1].round(3))
print(pd.crosstab(d.loc[te,'thermal_flag'],d.loc[te,'severity_code']))
print('---- follow-up')
dev=d.temp_cmd_deviation_C
for nme,msk in [('train single thermal',tr&(y==1)),('mixed thermal',te&(y==1))]:
    print(nme,'n',msk.sum(),'cmd dev: frac>0 %.2f mean %.1f'%((dev[msk]>0).mean(),dev[msk].mean()),' ext_temp-210 mean %.1f'%((d.ext_temp_mean_C[msk]-210).mean()))
A=Rn.abs()
lr3=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000)).fit(A[tr],y[tr]); report('F5 |residuals| + logistic',lr3.predict_proba(A[te])[:,1])
g3=HC(max_iter=150).fit(A[tr],y[tr]); report('F6 |residuals| + GBM',g3.predict_proba(A[te])[:,1])
Ar=d[raw].copy(); Ar['absdev']=(d.ext_temp_mean_C-210).abs()
g4=HC(max_iter=150).fit(Ar[tr],y[tr]); report('B3 raw + |T-210| GBM',g4.predict_proba(Ar[te])[:,1])
# small excursions: does residual beat B0 when |cmd dev|<=10?
sm=te&(dev.abs()<=10)&(y==1)|te&(y==0)
for nm_,s in [('B0',np.abs(d.ext_temp_mean_C-210)),('F1',np.abs(Rn['ext_temp_mean_C_res'])),('F5',pd.Series(lr3.predict_proba(A)[:,1],index=d.index))]:
    print('small-excursion subset',nm_,round(roc_auc_score(y[sm],s[sm]),3),'n_pos',int(y[sm].sum()))
# load effect size on temp in normal layers: how much does load explain?
print('ext_temp_mean std normal layers',d.loc[nm,'ext_temp_mean_C'].std().round(2),' residual std',Rn.loc[nm,'ext_temp_mean_C_res'].std().round(2))
