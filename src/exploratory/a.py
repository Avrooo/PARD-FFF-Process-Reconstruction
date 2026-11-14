import pandas as pd, numpy as np
d=pd.read_pickle('d.pkl'); p=pd.read_pickle('p.pkl')
try:
    import sklearn; print('sklearn',sklearn.__version__)
except Exception as e: print('no sklearn',e)
d['vol']=d.live_extrusion_positive_delta/d.layer_duration_s  # mm filament/s
# velocity ratio by speed cmd
print(d.groupby('speed_cmd_pct')[['velocity_ratio_mean','velocity_tracking_error_mean_mm_s']].agg(['mean','std','count']).round(3))
print(d[d.layer_id>0].groupby('speed_cmd_pct').velocity_ratio_mean.describe().round(3))
# power vs conditions (normal layers only)
n=d[(d.any_abnormal==0)&(d.layer_id>2)]
print(n[['ext_power_mean','vol','live_velocity_mean_mm_s','fan_actual_mean','ext_temp_std_C']].corr().round(2))
# power effect of flow/fan/speed/temp command
for c in ['temp_cmd_C','fan_cmd_norm','speed_cmd_pct','flow_cmd_pct']:
    print(c); print(d[d.layer_id>2].groupby(c)[['ext_power_mean','temp_error_mean_C','temp_error_std_C','vol','heater_on_fraction']].mean().round(3))
# per-fault means
print(d.groupby('fault_code')[['ext_power_mean','temp_error_mean_C','velocity_ratio_mean','vol','fan_actual_mean','bed_power_mean']].mean().round(3))
print(d.groupby('run_id').size().describe())
# does the temp deviation visible vs tracking? thermal flag 0 w/ dev
x=d[(d.temp_cmd_deviation_C!=0)&(d.thermal_flag==0)]; print(x[['temp_cmd_C','temp_cmd_deviation_C','severity_code','fault_code']].describe().round(2))
print(d.groupby('fault_code').severity_code.value_counts().unstack())
