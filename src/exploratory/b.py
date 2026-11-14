import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier as H
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
d=pd.read_pickle('d.pkl')
lab=['fault_code','any_abnormal','severity_code','thermal_flag','cooling_flag','speed_flag','underextrusion_flag']
cmd=[c for c in d.columns if 'cmd' in c or c in ('layer_z','planned_velocity_mean_mm_s','speed_factor_mean','extrude_factor_mean')]
sens=[c for c in d.columns if c not in lab+cmd+['run_id','layer_id']]
print('sensor feats',len(sens))
sens=[c for c in sens if c not in ('commanded_extrusion_positive_delta','extrusion_per_path_length','toolpath_length_xy')]  # commanded-ish
def cv(X,y,model):
    g=d.run_id; pr=np.zeros(len(d))
    for tr,te in GroupKFold(5).split(X,y,g):
        m=model(); m.fit(X.iloc[tr],y.iloc[tr]); pr[te]=m.predict_proba(X.iloc[te])[:,1]
    return roc_auc_score(y,pr)
base=['layer_id']
for t in ['any_abnormal','thermal_flag','cooling_flag','speed_flag','underextrusion_flag']:
    y=d[t]
    r=[cv(d[base],y,lambda:H(max_iter=100)), cv(d[sens],y,lambda:H(max_iter=150))]
    # + layer-relative features: deviation from run's own first normal-ish layers (layer 3-10 median)
    print(t,'AUC layer-only %.3f  sensor-only %.3f'%tuple(r))
# which fault flags are physically visible: effect sizes (normal vs flagged, layers>2)
n=d[d.layer_id>2]
for t in ['thermal_flag','cooling_flag','speed_flag','underextrusion_flag']:
    a=n[n[t]==1]; b=n[n.any_abnormal==0]
    z={c:(a[c].mean()-b[c].mean())/b[c].std() for c in sens}
    print(t,sorted(z.items(),key=lambda x:-abs(x[1]))[:5])
# is fan_actual just the command?
print(d[['fan_cmd_norm','fan_actual_mean']].corr().iloc[0,1], d[['temp_cmd_C','ext_temp_mean_C']].corr().iloc[0,1], d[['speed_cmd_pct','live_velocity_mean_mm_s']].corr().iloc[0,1])
print(d.groupby('layer_id')[['ext_temp_std_C','ext_power_mean','velocity_ratio_mean']].mean().iloc[[0,1,2,5,10,30,60]])
