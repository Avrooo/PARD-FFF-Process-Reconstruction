import warnings,sys; warnings.filterwarnings('ignore')
SEED=1
exec(open('base.py').read())
d=d.sort_values(['run_id','layer_id']).reset_index(drop=True)
d['rT']=d.ext_temp_mean_C-210
mixed=d.fault_type.str.startswith('Mixed'); tr=~mixed; te=mixed
# response of measured temp to commanded deviation, by magnitude bin and by layers since onset of the thermal block
x=d[te&(d.temp_cmd_deviation_C!=0)].copy()
x['blk']=(x.groupby('run_id').layer_id.diff()!=1).cumsum()
x['since']=x.groupby(['run_id','blk']).cumcount()
x['ratio']=x.rT/x.temp_cmd_deviation_C
print('mixed: measured/commanded by commanded-dev bin'); 
print(x.groupby(pd.cut(x.temp_cmd_deviation_C,[0,12,18,24,30,36])).agg(n=('rT','size'),cmd=('temp_cmd_deviation_C','mean'),meas=('rT','mean'),ratio=('ratio','median')).round(2))
print('by layers since onset of constant-looking block:'); print(x.groupby(x.since.clip(upper=6)).agg(n=('rT','size'),cmd=('temp_cmd_deviation_C','mean'),meas=('rT','mean'),ratio=('ratio','median')).round(2))
print('does cmd change within block (step vs ramp)?', x.groupby(['run_id','blk']).temp_cmd_deviation_C.agg(lambda s:s.nunique()).describe().round(2).to_dict())
t=d[tr&(d.temp_cmd_deviation_C!=0)].copy(); t['ratio']=t.rT/t.temp_cmd_deviation_C
print('single-fault cold: median meas/cmd %.2f'%t.ratio.median(),' power mean',t.ext_power_mean.mean().round(2),'| mixed hot power mean',x.ext_power_mean.mean().round(2),' heater_on',x.heater_on_fraction.mean().round(3),' power_max',x.ext_power_max.mean().round(2))
print('hot excursions in mixed, power_max>=0.99 fraction:',(x.ext_power_max>=0.99).mean().round(2))
