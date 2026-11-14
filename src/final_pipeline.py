"""PARD final pipeline: regenerates every number, table and figure in the manuscript.
Run from repository root: python src/final_pipeline.py (requires input files in data/ and running src/prof.py first)"""
import warnings,json,glob,sys; warnings.filterwarnings('ignore')
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
GEN = ROOT / 'gen'
GEN.mkdir(exist_ok=True)
import os
os.chdir(ROOT)
files_xlsx = sorted(DATA.glob('*.xlsx'))
if len(files_xlsx) != 1:
    raise FileNotFoundError('Place exactly one telemetry .xlsx in data/')
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor as HR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
UP=str(DATA) + os.sep
d0=pd.read_pickle(GEN / 'd.pkl'); p=pd.read_pickle(GEN / 'p.pkl')
ra=pd.read_excel(str(files_xlsx[0]),sheet_name='Run_Audit')[['run_id','fault_type','schedule_family','abnormal_blocks']]
geo=['base_path_length_xy_mm','base_extrusion_path_xy_mm','base_travel_path_xy_mm','move_count','direction_change_count','sharp_turns_per_100mm','perimeter_path_fraction','infill_path_fraction','skin_path_fraction']
d=d0.merge(ra[['run_id','fault_type','schedule_family']],on='run_id').merge(p[['run_id','layer_id','base_nominal_extrusion_mm','base_nominal_layer_duration_s']+geo],on=['run_id','layer_id'])
R={}   # results store
R['layers_total']=len(d); R['prints_total']=d.run_id.nunique()
d=d[d.layer_duration_s>0.5]; R['layers_degenerate']=R['layers_total']-len(d)
d=d.sort_values(['run_id','layer_id']).reset_index(drop=True)
mixed=d.fault_type.str.startswith('Mixed'); tr=~mixed; te=mixed
R['layers_used']=len(d); R['layers_dev']=int(tr.sum()); R['layers_test']=int(te.sum())
R['prints_dev']=d.run_id[tr].nunique(); R['prints_test']=d.run_id[te].nunique()
ft=ra.set_index('run_id').fault_type; R['print_counts']=ft.value_counts().to_dict()
R['schedule_counts']=ra.schedule_family.value_counts().to_dict()
R['sampling_hz_median']=float((d.samples_in_layer/d.layer_duration_s).median()); R['layer_duration_median_s']=float(d.layer_duration_s.median())
# ---- plan-referenced residuals (plan = original G-code: 210 C, fan 1, 100%, 100%)
d['rT']=d.ext_temp_mean_C-210; d['rF']=d.fan_actual_mean-1.0
d['rS']=(d.base_nominal_layer_duration_s/d.layer_duration_s-1)*100
d['rE']=(d.live_extrusion_positive_delta/d.base_nominal_extrusion_mm-1)*100
res={'thermal':'rT','cooling':'rF','speed':'rS','flow':'rE'}
Y={'thermal':'temp_cmd_deviation_C','cooling':'fan_cmd_deviation_norm','speed':'speed_cmd_deviation_pct','flow':'flow_cmd_deviation_pct'}
for c,r in res.items():
    for L in (1,2):
        d[f'{r}_n{L}']=d.groupby('run_id')[r].shift(-L); d[f'{r}_n{L}']=d[f'{r}_n{L}'].fillna(d[r])
EXC=['run_id','layer_id','fault_code','any_abnormal','severity_code','thermal_flag','cooling_flag','speed_flag','underextrusion_flag','fault_type','schedule_family','layer_z','planned_velocity_mean_mm_s','commanded_extrusion_positive_delta','extrusion_per_path_length','speed_factor_mean','extrude_factor_mean','base_nominal_extrusion_mm','base_nominal_layer_duration_s']
raw=[c for c in d0.columns if c not in EXC and 'cmd' not in c]; R['n_raw_features']=len(raw); R['raw_features']=raw
allres=['rT','rF','rS','rE']; TOL={'thermal':5.0,'cooling':0.2,'speed':10.0,'flow':10.0}
def lin(f,m,y): mm=LinearRegression().fit(d.loc[m,f],d.loc[m,y]); return mm.predict(d[f])
def gbm(f,m,y): mm=HR(max_iter=150).fit(d.loc[m,f],d.loc[m,y]); return mm.predict(d[f])
# ---------- (1) development-only estimator routing: run-grouped CV, simplest within 10% of best (order C1<C2<C3)
CANDS=['C1','C2','C3']; NAMES={'C1':'Same-layer linear','C2':'Delayed linear','C3':'Learned'}
dev_idx=np.where(tr.values)[0]; runs_dev=d.run_id.values[dev_idx]
def feats(c,k): r=res[c]; return {'C1':[r],'C2':[r+'_n1'],'C3':raw+[r]}[k]
CV={}; PICK={}
for c in Y:
    y=Y[c]; CV[c]={}
    for k in CANDS:
        err=[]
        for a,b in GroupKFold(5).split(dev_idx,groups=runs_dev):
            ia,ib=dev_idx[a],dev_idx[b]; ea=ia[d[y].values[ia]!=0]; eb=ib[d[y].values[ib]!=0]
            m=(HR(max_iter=150).fit(d.iloc[ia][feats(c,k)],d[y].iloc[ia]) if k=='C3' else LinearRegression().fit(d.iloc[ea][feats(c,k)],d[y].iloc[ea]))
            err.append(np.abs(d[y].values[eb]-m.predict(d.iloc[eb][feats(c,k)])))
        CV[c][k]=float(np.concatenate(err).mean())
    best=min(CV[c].values()); PICK[c]=[k for k in CANDS if CV[c][k]<=1.10*best][0]
R['cv']=CV; R['pick']=PICK
# ---------- helpers
def gate(pr,c,cal):
    nc=cal&(d[Y[c]]==0); thr=np.quantile(np.abs(pr[nc.values]),.99); g=np.abs(pr)>thr; return np.where(g,pr,0.0),g,thr
def fit_c(c,k,proper):
    y=Y[c]; return gbm(feats(c,k),proper,y) if k=='C3' else lin(feats(c,k),proper&(d[y]!=0),y)
nomdev=tr&(d.any_abnormal==0)&(d.layer_id>2)
TOLB={c:float(3*d.loc[nomdev,res[c]].std()) for c in Y}; R['tolB']=TOLB
METHODS=['B1','B2','B3','PARD']
SEEDS=[1,2,3,4,5]; per={m:[] for m in METHODS}; chan_rows=[]; abl_rows=[]; rel_rows=[]; sens_rows=[]
store1={}
rid=d.run_id.values; tm=te.values
def metrics(lab,S,tol=None):
    return dict(AUC=roc_auc_score(lab[tm],S[tm]),F1=f1_score(lab[tm],S[tm]>1),prec=precision_score(lab[tm],S[tm]>1),rec=recall_score(lab[tm],S[tm]>1))
for seed in SEEDS:
    runs=d.run_id[tr].unique(); rng=np.random.default_rng(seed); rng.shuffle(runs)
    cal_runs=set(runs[:int(.3*len(runs))]); proper=tr&~d.run_id.isin(cal_runs); cal=tr&d.run_id.isin(cal_runs)
    P={m:{} for m in METHODS}
    for c in Y:
        y=Y[c]; r=res[c]; exm=proper&(d[y]!=0)
        P['B1'][c]=d[r].values
        P['B2'][c]=lin([r],proper if c=='thermal' else exm,y)
        P['B3'][c]=gbm(raw+geo+allres,proper,y)
        P['PARD'][c]=fit_c(c,PICK[c],proper)
    PG={};G={};TH={}
    for m in METHODS:
        for c in Y: PG[(m,c)],G[(m,c)],TH[(m,c)]=gate(P[m][c],c,cal)
    def Sscore(m,tol): return np.max([np.abs(PG[(m,c)])/tol[c] for c in Y],axis=0)
    for tname,tol in [('A',TOL),('Ax0.5',{c:v*.5 for c,v in TOL.items()}),('Ax2',{c:v*2 for c,v in TOL.items()}),('B',TOLB)]:
        Strue=np.max([np.abs(d[Y[c]].values)/tol[c] for c in Y],axis=0); lab=Strue>1
        for m in METHODS:
            S=Sscore(m,tol); mt=metrics(lab,S); mt['Smae']=float(np.mean(np.abs(S[tm]-Strue[tm]))); mt.update(seed=seed,tol=tname,method=m,n_oot=int(lab[tm].sum()))
            sens_rows.append(mt)
        if tname=='A' and seed==1:
            store1['lab']=lab; store1['S']={m:Sscore(m,tol) for m in METHODS}; store1['Strue']=Strue
    # per-channel excursion-layer MAE, gate stats, coverage for PARD and comparators
    for c in Y:
        y=Y[c]; yt=d[y].values; x=(yt!=0)&tm
        for m in METHODS:
            pr=P[m][c]; g=G[(m,c)]
            # conformal (excursion calibration layers that pass the gate)
            ce=(cal&(d[y]!=0)).values&g; errs=np.abs(d[y].values-pr)[ce]
            q=np.quantile(errs,.9)*(1+1/len(errs)) if len(errs)>5 else np.nan
            e=np.abs(yt-pr)
            chan_rows.append(dict(seed=seed,chan=c,method=m,MAE=e[x].mean(),recall=g[x].mean(),FAR=g[tm&(yt==0)].mean(),
                                  cover=(np.mean(e[x&g]<=q) if (x&g).any() and q==q else np.nan),width=2*q))
        # delay ablation for every channel (excursion-fit linear): L0,L1,L2 and learned
        for k,lbl in [('C1','delay0'),('C2','delay1'),('C3','learned')]:
            pr=fit_c(c,k,proper); abl_rows.append(dict(seed=seed,chan=c,variant=lbl,MAE=np.abs(yt-pr)[x].mean()))
        r=res[c]; pr2=lin([r+'_n2'],proper&(d[y]!=0),y); abl_rows.append(dict(seed=seed,chan=c,variant='delay2',MAE=np.abs(yt-pr2)[x].mean()))
    if seed==1: store1['P']=P; store1['G']=G; store1['PG']=PG
# ---------- aggregate
CR=pd.DataFrame(chan_rows); AB=pd.DataFrame(abl_rows); SR=pd.DataFrame(sens_rows)
R['sens']=SR.groupby(['tol','method']).agg(AUC=('AUC','mean'),AUCsd=('AUC','std'),F1=('F1','mean'),F1sd=('F1','std'),prec=('prec','mean'),rec=('rec','mean'),Smae=('Smae','mean'),Smaesd=('Smae','std'),n_oot=('n_oot','first')).round(4).reset_index().to_dict('records')
R['chan']=CR.groupby(['chan','method']).agg(MAE=('MAE','mean'),MAEsd=('MAE','std'),recall=('recall','mean'),FAR=('FAR','mean'),cover=('cover','mean'),cover_min=('cover','min'),width=('width','mean')).round(4).reset_index().to_dict('records')
R['abl']=AB.groupby(['chan','variant']).agg(MAE=('MAE','mean'),MAEsd=('MAE','std')).round(4).reset_index().to_dict('records')
# ---------- paired run-clustered bootstrap (seed-1 split, tolerance set A)
lab=store1['lab']; S=store1['S']; ur=np.unique(rid[tm]); ix={r:np.where(rid==r)[0] for r in ur}; rng=np.random.default_rng(0); B=2000
pairs=[('PARD','B3'),('PARD','B2'),('PARD','B1')]; D={p_:{'AUC':[],'F1':[],'Smae':[]} for p_ in pairs}
for _ in range(B):
    ii=np.concatenate([ix[r] for r in rng.choice(ur,len(ur))])
    for a,b in pairs:
        D[(a,b)]['AUC'].append(roc_auc_score(lab[ii],S[a][ii])-roc_auc_score(lab[ii],S[b][ii]))
        D[(a,b)]['F1'].append(f1_score(lab[ii],S[a][ii]>1)-f1_score(lab[ii],S[b][ii]>1))
        D[(a,b)]['Smae'].append(np.mean(np.abs(S[a][ii]-store1['Strue'][ii]))-np.mean(np.abs(S[b][ii]-store1['Strue'][ii])))
R['paired']={f'{a}-{b}':{k:[float(np.mean(v)),float(np.percentile(v,2.5)),float(np.percentile(v,97.5))] for k,v in D[(a,b)].items()} for a,b in pairs}
# ---------- event statistics (true excursion blocks)
ev=[]
for c in Y:
    y=d[Y[c]].values
    for r in np.unique(rid):
        ixr=np.where(rid==r)[0]; yy=y[ixr]; k=np.where(yy!=0)[0]
        if len(k)==0: continue
        sp=np.where((np.diff(k)!=1)|(np.sign(yy[k][1:])!=np.sign(yy[k][:-1])))[0]+1
        for blk in np.split(k,sp): ev.append(dict(chan=c,run=r,mixed=bool(mixed.values[ixr[0]]),len=len(blk),start=int(d.layer_id.values[ixr[blk[0]]]),mean_dev=float(yy[blk].mean())))
EV=pd.DataFrame(ev); R['events_mixed']=EV[EV.mixed].groupby('chan').agg(n=('len','size'),len_med=('len','median'),len_min=('len','min'),len_max=('len','max'),start_min=('start','min')).to_dict('index')
R['events_dev']=EV[~EV.mixed].groupby('chan').agg(n=('len','size'),len_med=('len','median'),len_min=('len','min'),len_max=('len','max'),mean_dev=('mean_dev','mean')).to_dict('index')
# direction/magnitude ranges of true deviations (dev vs mixed)
rng_rows={}
for c in Y:
    a=d.loc[tr&(d[Y[c]]!=0),Y[c]]; b=d.loc[te&(d[Y[c]]!=0),Y[c]]
    rng_rows[c]=dict(dev_n=int(len(a)),dev_min=float(a.min()),dev_max=float(a.max()),dev_pos=float((a>0).mean()),mix_n=int(len(b)),mix_min=float(b.min()),mix_max=float(b.max()),mix_pos=float((b>0).mean()))
R['dev_ranges']=rng_rows
# thermal step response by layers since onset (development single-fault prints and mixed test prints)
def onset_curve(mask):
    rows=[]; y=d[Y['thermal']].values; rT=d.rT.values
    for r in np.unique(rid[mask.values]):
        ixr=np.where(rid==r)[0]; yy=y[ixr]; k=np.where(yy!=0)[0]
        if len(k)==0: continue
        sp=np.where((np.diff(k)!=1)|(np.sign(yy[k][1:])!=np.sign(yy[k][:-1])))[0]+1
        for blk in np.split(k,sp):
            for j,i in enumerate(blk[:6]):
                if abs(yy[i])>=5: rows.append(dict(since=j,ratio=rT[ixr[i]]/yy[i]))
    df=pd.DataFrame(rows); return df.groupby('since').ratio.agg(['median','count']).reset_index()
R['onset_dev']=onset_curve(tr).round(3).to_dict('records'); R['onset_mixed']=onset_curve(te).round(3).to_dict('records')
json.dump(R,open('gen/results.json','w'),indent=1,default=float)
pd.to_pickle(dict(store1=store1,d=d,Y=Y,res=res,TOL=TOL,PICK=PICK,R=R),'gen/store.pkl')
print('done',R['layers_used'],R['pick'])
