import pandas as pd, numpy as np, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "gen"
OUT.mkdir(exist_ok=True)
files_xlsx = sorted(DATA.glob("*.xlsx"))
files_csv = sorted(DATA.glob("*.csv"))
if len(files_xlsx) != 1 or len(files_csv) != 1:
    raise FileNotFoundError("Place exactly one telemetry .xlsx and one planned G-code .csv in data/")
f=files_xlsx[0]
d=pd.read_excel(f,sheet_name='Training_Data'); d.to_pickle(OUT / 'd.pkl')
p=pd.read_csv(files_csv[0]); p.to_pickle(OUT / 'p.pkl')
print(d.shape,p.shape)
print(d.fault_code.value_counts()); print(d.severity_code.value_counts())
print(d[['thermal_flag','cooling_flag','speed_flag','underextrusion_flag']].mean())
# are flags deterministic from commands?
print(pd.crosstab(d.thermal_flag,d.temp_cmd_deviation_C!=0))
print(pd.crosstab(d.cooling_flag,d.fan_cmd_deviation_norm!=0))
print(pd.crosstab(d.speed_flag,d.speed_cmd_deviation_pct!=0))
print(pd.crosstab(d.underextrusion_flag,d.flow_cmd_deviation_pct!=0))
print(pd.crosstab(d.underextrusion_flag,d.flow_cmd_deviation_pct.apply(lambda x:'neg' if x<0 else('pos' if x>0 else 0))))
print(d[['ext_temp_mean_C','temp_error_mean_C','ext_power_mean','heater_on_fraction','live_velocity_mean_mm_s','planned_velocity_mean_mm_s','velocity_ratio_mean','live_extrusion_positive_delta','commanded_extrusion_positive_delta','layer_duration_s']].describe().T)
print(p.plan_source.value_counts()); print(p.columns.tolist()[:50])
