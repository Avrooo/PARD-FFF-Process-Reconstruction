import warnings,sys,json; warnings.filterwarnings('ignore')
src=open('i.py').read().split("cfg={'thermal'")[0]
exec(src)
for L in (1,2):
    d[f'rT_n{L}']=d.groupby('run_id').rT.shift(-L); d[f'rT_n{L}']=d[f'rT_n{L}'].fillna(d.rT)
d['rT_n1']=d['rT_n1']; 
M={}
M[('thermal','L0 now (real-time)')]=('lin',['rT'],proper,'thermal')
M[('thermal','L1 one-layer delay')]=('lin',['rT_n1'],proper,'thermal')
M[('thermal','L2 two-layer delay')]=('lin',['rT_n2'],proper,'thermal')
M[('thermal','L1+heater power')]=('lin',['rT_n1','rT_pow'],proper,'thermal')
M[('thermal','learned raw+res (L0)')]=('gbm',raw+['rT','rT_prev'],proper,'thermal')
M[('cooling','plan-ref (excursion OLS)')]=('lin',['rF'],proper&(d.cooling_flag==1),'cooling')
M[('cooling','learned raw+res')]=('gbm',raw+['rF'],proper,'cooling')
M[('speed','learned raw+res')]=('gbm',raw+['rS'],proper,'speed')
M[('speed','learned raw (no residual)')]=('gbm',raw,proper,'speed')
M[('flow','learned raw+res')]=('gbm',raw+['rE'],proper,'flow')
M[('flow','learned raw (no residual)')]=('gbm',raw,proper,'flow')
runs_te=d.loc[te,'run_id'].values; ur=np.unique(runs_te)
rng=np.random.default_rng(0); B=1000
rows=[]
for (k,nm),(kind,feats,fm,_) in M.items():
    y=ch[k][1]; fn=mk(kind,feats,fm,y); R=evaluate(fn,y)
    # cal-set MAE on (cold) excursion layers, for pre-specified selection
    ce=cal&(d[y]!=0); calmae=np.abs(d.loc[ce,y]-fn(ce)).mean()
    e,x,g,q=R['e'],R['x'],R['g'],R['q']
    idx={r:np.where(runs_te==r)[0] for r in ur}
    bm=[];bc=[]
    for _ in range(B if SEED==1 else 0):
        pick=rng.choice(ur,len(ur)); ii=np.concatenate([idx[r] for r in pick])
        xx=x[ii]; gg=g[ii]; ee=e[ii]
        bm.append(ee[xx].mean()); bc.append(np.mean(ee[xx&gg]<=q) if (xx&gg).any() else np.nan)
    ci=lambda a:(np.nanpercentile(a,2.5),np.nanpercentile(a,97.5)) if len(a) else (np.nan,np.nan)
    rows.append(dict(seed=SEED,chan=k,method=nm,MAE=R['MAEexc'],recall=R['recall'],FAR=R['FAR'],cover=R['cover'],calMAE=calmae,bias=R['bias_big'],mae_ci=ci(bm),cov_ci=ci(bc)))
json.dump(rows,open(f'res_{SEED}.json','w'),default=float)
