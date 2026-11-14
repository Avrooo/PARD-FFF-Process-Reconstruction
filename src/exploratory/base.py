import pandas as pd, numpy as np, warnings, glob; warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingRegressor as HR
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import roc_auc_score
d=pd.read_pickle('d.pkl'); p=pd.read_pickle('p.pkl')
ra=pd.read_excel(glob.glob('/mnt/user-data/uploads/*.xlsx')[0],sheet_name='Run_Audit')[['run_id','fault_type','schedule_family']]
d=d.merge(ra,on='run_id').merge(p[['run_id','layer_id','base_nominal_extrusion_mm','base_nominal_layer_duration_s']],on=['run_id','layer_id'])
d=d[d.layer_duration_s>0.5].reset_index(drop=True)
mixed=d.fault_type.str.startswith('Mixed'); tr=~mixed; te=mixed
# plan-referenced, sign-aware residuals (reference = original G-code plan: 210C, fan 1, 100%, 100%)
d['rT']=d.ext_temp_mean_C-210
d['rF']=d.fan_actual_mean-1.0
d['rS']=(d.base_nominal_layer_duration_s/d.layer_duration_s-1)*100
d['rE']=(d.live_extrusion_positive_delta/d.base_nominal_extrusion_mm-1)*100
ch={'thermal':('rT','temp_cmd_deviation_C','thermal_flag'),'cooling':('rF','fan_cmd_deviation_norm','cooling_flag'),
    'speed':('rS','speed_cmd_deviation_pct','speed_flag'),'flow':('rE','flow_cmd_deviation_pct','underextrusion_flag')}
for k,(r,y,f) in ch.items():
    a=d[tr&(d[y]!=0)][y]; b=d[te&(d[y]!=0)][y]
    print(f'{k:8s} true-dev  train: n={len(a)} frac>0={np.mean(a>0):.2f} mean={a.mean():.2f} range[{a.min():.1f},{a.max():.1f}] | mixed: n={len(b)} frac>0={np.mean(b>0):.2f} mean={b.mean():.2f} range[{b.min():.1f},{b.max():.1f}]')
raw=[c for c in d.columns if c not in ['run_id','layer_id','fault_code','any_abnormal','severity_code','thermal_flag','cooling_flag','speed_flag','underextrusion_flag','fault_type','schedule_family','layer_z','planned_velocity_mean_mm_s','commanded_extrusion_positive_delta','extrusion_per_path_length','speed_factor_mean','extrude_factor_mean','base_nominal_extrusion_mm','base_nominal_layer_duration_s','rT','rF','rS','rE'] and 'cmd' not in c]
runs=d.run_id[tr].unique(); rng=np.random.default_rng(SEED); rng.shuffle(runs)
cal_runs=set(runs[:int(.3*len(runs))]); proper=tr&~d.run_id.isin(cal_runs); cal=tr&d.run_id.isin(cal_runs)
