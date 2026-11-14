import warnings,sys; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 1
exec(open('base.py').read())
# --- lag / dynamic thermal features (all computable online from telemetry + plan)
d=d.sort_values(['run_id','layer_id']).reset_index(drop=True)
mixed=d.fault_type.str.startswith('Mixed'); tr=~mixed; te=mixed
runs=d.run_id[tr].unique(); rng=np.random.default_rng(SEED); rng.shuffle(runs)
cal_runs=set(runs[:int(.3*len(runs))]); proper=tr&~d.run_id.isin(cal_runs); cal=tr&d.run_id.isin(cal_runs)
d['rT']=d.ext_temp_mean_C-210
d['rT_prev']=d.groupby('run_id').rT.shift(1).fillna(d.rT)
d['rT_prev2']=d.groupby('run_id').rT.shift(2).fillna(d.rT_prev)
d['rT_max']=d.ext_temp_max_C-210; d['rT_min']=d.ext_temp_min_C-210
d['rT_end']=d.rT+d.ext_temp_slope_C_per_s*d.layer_duration_s/2   # extrapolate mean to end of layer by slope
d['rT_pow']=d.ext_power_mean   # heater effort: a saturated heater means temp understates command
raw=[c for c in raw if c in d.columns]
alpha=.1
def mk(kind,feats,fit_mask,y):
    if kind=='lin': m=LinearRegression().fit(d.loc[fit_mask,feats],d.loc[fit_mask,y]); return lambda msk:m.predict(d.loc[msk,feats])
    m=HR(max_iter=150).fit(d.loc[fit_mask,feats],d.loc[fit_mask,y]); return lambda msk:m.predict(d.loc[msk,feats])
def evaluate(fn,y,gate_frac=.99):
    nc=cal&(d[y]==0); thr=np.quantile(np.abs(fn(nc)),gate_frac)
    pc_all=pd.Series(fn(cal),index=d.index[cal]); ce=(d.loc[cal,y]!=0)&(pc_all.abs()>thr)
    errs=(d.loc[cal,y]-pc_all).abs()[ce]
    q=np.quantile(errs,1-alpha)*(1+1/len(errs)) if len(errs)>5 else np.nan
    pt=fn(te); yt=d.loc[te,y].values; e=np.abs(yt-pt); x=yt!=0; g=np.abs(pt)>thr
    big=x&(np.abs(yt)>d.loc[tr,y].abs().max())
    return dict(MAEexc=e[x].mean(),MAEexc_gated=e[x&g].mean() if (x&g).any() else np.nan,recall=g[x].mean(),FAR=g[~x].mean(),
                cover=np.mean(e[x&g]<=q) if (x&g).any() else np.nan,width=2*q,bias_big=(pt[big]-yt[big]).mean() if big.any() else np.nan,
                pred=pt,e=e,x=x,g=g,q=q)
cfg={'thermal':{ 'A0 plan-ref (rT)':('lin',['rT'],proper),
                 'A1 +lag (rT,rT_prev)':('lin',['rT','rT_prev'],proper),
                 'A2 +lag2':('lin',['rT','rT_prev','rT_prev2'],proper),
                 'A3 +end-of-layer slope':('lin',['rT','rT_end'],proper),
                 'A4 +min/max':('lin',['rT','rT_max','rT_min'],proper),
                 'A5 +lag+heater power':('lin',['rT','rT_prev','rT_pow'],proper),
                 'A6 learned raw':('gbm',raw,proper),
                 'A7 learned raw+res':('gbm',raw+['rT','rT_prev'],proper)}}
for k,(r,y,f) in ch.items():
    if k=='thermal': continue
    exm=proper&(d[y]!=0)
    cfg[k]={'P plan-ref (all-layer OLS)':('lin',[r],proper),'P plan-ref (excursion OLS)':('lin',[r],exm),
            'L learned raw':('gbm',raw,proper),'L learned raw+res':('gbm',raw+[r],proper)}
out={}
for k,(r,y,f) in ch.items():
    print('\n==',k,' (excursion layers in mixed test:',int((d.loc[te,y]!=0).sum()),')')
    print(f"{'variant':30s}{'MAEexc':>8s}{'gated':>8s}{'recall':>8s}{'FAR':>7s}{'cover':>7s}{'width':>8s}{'bias>trmax':>11s}")
    for nm,(kind,feats,fm) in cfg[k].items():
        fn=mk(kind,feats,fm,y); R=evaluate(fn,y); out[(k,nm)]=R
        print(f"{nm:30s}{R['MAEexc']:8.2f}{R['MAEexc_gated']:8.2f}{R['recall']:8.2f}{R['FAR']:7.3f}{R['cover']:7.2f}{R['width']:8.2f}{R['bias_big']:11.2f}")
import pickle; pickle.dump({k:{a:b for a,b in v.items() if a in('pred','e','x','g','q')} for k,v in out.items()},open(f'out_{SEED}.pkl','wb'))
d.loc[te,['run_id']].to_pickle(f'runs_{SEED}.pkl')
