import warnings,sys; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 1
BOOT=(SEED==1 and len(sys.argv)<=2)
exec(open('base.py').read())
geo=['base_path_length_xy_mm','base_extrusion_path_xy_mm','base_travel_path_xy_mm','move_count','direction_change_count','sharp_turns_per_100mm','perimeter_path_fraction','infill_path_fraction','skin_path_fraction']
p=pd.read_pickle('p.pkl')
d=d.merge(p[['run_id','layer_id']+geo],on=['run_id','layer_id'],how='left').sort_values(['run_id','layer_id']).reset_index(drop=True)
mixed=d.fault_type.str.startswith('Mixed'); tr=~mixed; te=mixed
runs=d.run_id[tr].unique(); rng=np.random.default_rng(SEED); rng.shuffle(runs)
cal_runs=set(runs[:int(.3*len(runs))]); proper=tr&~d.run_id.isin(cal_runs); cal=tr&d.run_id.isin(cal_runs)
d['rT']=d.ext_temp_mean_C-210
d['rT_n1']=d.groupby('run_id').rT.shift(-1); d['rT_n1']=d.rT_n1.fillna(d.rT)
res={'thermal':'rT','cooling':'rF','speed':'rS','flow':'rE'}
Y={'thermal':'temp_cmd_deviation_C','cooling':'fan_cmd_deviation_norm','speed':'speed_cmd_deviation_pct','flow':'flow_cmd_deviation_pct'}
TS=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
TOL={'thermal':5.0*TS,'cooling':0.2*TS,'speed':10.0*TS,'flow':10.0*TS}   # fixed a priori engineering tolerances, NOT tuned
raw=[c for c in raw if c in d.columns and c not in geo]
allres=['rT','rF','rS','rE']
def lin(f,m,y): mm=LinearRegression().fit(d.loc[m,f],d.loc[m,y]); return mm.predict(d[f])
def gbm(f,m,y): mm=HR(max_iter=150).fit(d.loc[m,f],d.loc[m,y]); return mm.predict(d[f])
P={}   # P[method][channel] = prediction over all rows
for c in Y:
    y=Y[c]; r=res[c]; exm=proper&(d[y]!=0)
    P.setdefault('B1 same-layer residual (untuned)',{})[c]=d[r].values
    P.setdefault('B2 tuned per-channel residual',{})[c]=lin([r],proper if c=='thermal' else exm,y)
    P.setdefault('B3 learned, plan-aware (same layer)',{})[c]=gbm(raw+geo+allres,proper,y)
P['PARD']={'thermal':lin(['rT_n1'],proper,Y['thermal']),
           'cooling':lin(['rF'],proper&(d[Y['cooling']]!=0),Y['cooling']),
           'speed':gbm(raw+['rS'],proper,Y['speed']),'flow':gbm(raw+['rE'],proper,Y['flow'])}
# gates (99th pct of nominal calibration layers), gated predictions
G={};PG={}
for m,pc in P.items():
    for c,pr in pc.items():
        nc=cal&(d[Y[c]]==0); thr=np.quantile(np.abs(pr[nc.values]),.99)
        G[(m,c)]=np.abs(pr)>thr; PG[(m,c)]=np.where(G[(m,c)],pr,0.0)
tm=te.values; rid=d.run_id.values; lid=d.layer_id.values
# ---- unified execution-fidelity score S = max_c |dev_c|/tol_c
Strue=np.max([np.abs(d[Y[c]].values)/TOL[c] for c in Y],axis=0); lab=(Strue>1)
rows=[]; ev=[]
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
def runs_idx(): return {r:np.where((rid==r)&tm)[0] for r in np.unique(rid[tm])}
RI=runs_idx()
for m in P:
    Sh=np.max([np.abs(PG[(m,c)])/TOL[c] for c in Y],axis=0)
    rows.append(dict(method=m,AUC=roc_auc_score(lab[tm],Sh[tm]),F1=f1_score(lab[tm],Sh[tm]>1),prec=precision_score(lab[tm],Sh[tm]>1),rec=recall_score(lab[tm],Sh[tm]>1),
                     S_MAE=np.mean(np.abs(Sh[tm]-Strue[tm])),S=Sh))
    # ---- event-level
    for c in Y:
        y=d[Y[c]].values; pr=P[m] [c]; g=G[(m,c)]
        for r,ix in RI.items():
            yy=y[ix]; k=np.where(yy!=0)[0]
            if len(k)==0: continue
            # events: consecutive layers, same sign
            splits=np.where((np.diff(k)!=1)|(np.sign(yy[k][1:])!=np.sign(yy[k][:-1])))[0]+1
            for blk in np.split(k,splits):
                ii=ix[blk]; lo,hi=max(ix[blk[0]]-1,ix[0]),min(ix[blk[-1]]+1,ix[-1])
                win=np.arange(lo,hi+1)
                ev.append(dict(method=m,chan=c,run=r,recovered=g[ii].mean()>=.5,sign_ok=np.sign(pr[ii].mean())==np.sign(y[ii].mean()),
                               relerr=abs(pr[ii].mean()-y[ii].mean())/abs(y[ii].mean()),dur_err=g[win].sum()-len(ii),dur_true=len(ii)))
    # false events: >=2 consecutive gated layers on nominal layers of a channel (per print)
    for c in Y:
        g=G[(m,c)]; y=d[Y[c]].values; fe={}
        for r,ix in RI.items():
            bad=(g[ix])&(y[ix]==0); n=0;run=0
            for b in bad:
                run=run+1 if b else 0
                if run==2: n+=1
            fe[r]=n
        rows.append(dict(method=m,chan=c,false_events_per_print=np.mean(list(fe.values())),fe=fe))
S_rows=[r for r in rows if 'AUC' in r]; F_rows=[r for r in rows if 'fe' in r]
EV=pd.DataFrame(ev)
print(f'SEED {SEED}')
print('\nUNIFIED EXECUTION-FIDELITY SCORE  S=max_c |dev_c|/tol_c  (tol: thermal 5C, fan 0.2, speed 10%%, flow 10%%); out-of-tolerance layers: %d of %d mixed test layers'%(lab[tm].sum(),tm.sum()))
print(pd.DataFrame(S_rows).drop(columns='S').round(3).to_string(index=False))
print('\nEVENT-LEVEL (20 mixed prints; event = contiguous true excursion block)')
agg=EV.groupby(['chan','method']).agg(events=('run','size'),recovered=('recovered','mean'),sign_ok=('sign_ok','mean'),med_rel_mag_err=('relerr','median'),mean_dur_err=('dur_err','mean')).round(3)
print(agg.to_string())
fe=pd.DataFrame(F_rows).drop(columns='fe').pivot(index='chan',columns='method',values='false_events_per_print').round(2)
print('\nFALSE EVENTS per print (>=2 consecutive gated nominal layers):'); print(fe.to_string())
EV.to_csv(f'events_seed{SEED}.csv',index=False)
if BOOT:
    rng=np.random.default_rng(0); ur=np.unique(rid[tm]); B=1000; out=[]
    S_by={r['method']:r['S'] for r in S_rows}
    ixr={r:np.where((rid==r))[0] for r in ur}; evg={k:v for k,v in EV.groupby(['chan','method'])}
    boot={}
    for _ in range(B):
        pick=rng.choice(ur,len(ur)); ii=np.concatenate([ixr[r] for r in pick])
        for m in P:
            if lab[ii].min()==lab[ii].max(): continue
            boot.setdefault((m,'AUC'),[]).append(roc_auc_score(lab[ii],S_by[m][ii]))
            boot.setdefault((m,'F1'),[]).append(f1_score(lab[ii],S_by[m][ii]>1))
        cnt=pd.Series(pick).value_counts()
        for (c,m),e in evg.items():
            w=e.run.map(cnt).fillna(0).values
            if w.sum()==0: continue
            boot.setdefault((m,c,'recovered'),[]).append(np.average(e.recovered,weights=w))
            boot.setdefault((m,c,'relerr'),[]).append(np.average(e.relerr,weights=w))
    print('\nBOOTSTRAP 95% CI over runs (1000 resamples, seed-1 split)')
    for k,v in boot.items():
        if len(k)==2 or k[0] in ('PARD','B1 same-layer residual (untuned)','B2 tuned per-channel residual','B3 learned, plan-aware (same layer)'):
            print(k, '%.3f [%.3f, %.3f]'%(np.mean(v),*np.percentile(v,[2.5,97.5])))
