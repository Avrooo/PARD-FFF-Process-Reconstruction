import warnings,sys; warnings.filterwarnings('ignore')
SEED=int(sys.argv[1]) if len(sys.argv)>1 else 1
sys.argv=[sys.argv[0],str(SEED)]
exec(open('m.py').read().split("# ---- unified execution-fidelity score")[0])
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, f1_score
for c in res: 
    r=res[c]; d[r+'_n1']=d.groupby('run_id')[r].shift(-1); d[r+'_n1']=d[r+'_n1'].fillna(d[r])
def gate_pred(pr,c):
    nc=cal&(d[Y[c]]==0); thr=np.quantile(np.abs(pr[nc.values]),.99); g=np.abs(pr)>thr; return np.where(g,pr,0.0),g
def score(PGd,tol): return np.max([np.abs(PGd[c])/tol[c] for c in Y],axis=0)
# ---- (B) routing chosen ONLY from development prints (single-fault + normal), grouped CV by run
dev=tr; runs_dev=d.run_id[dev].values
CANDS=['C1 same-layer linear','C2 one-layer-delay linear','C3 learned (raw+residual)']
def cv_mae(c,cand):
    y=Y[c]; r=res[c]; X={'C1 same-layer linear':[r],'C2 one-layer-delay linear':[r+'_n1'],'C3 learned (raw+residual)':raw+[r]}[cand]
    idx=np.where(dev.values)[0]; err=[]
    for a,b in GroupKFold(5).split(idx,groups=runs_dev):
        ia,ib=idx[a],idx[b]; ea=ia[d[y].values[ia]!=0]; eb=ib[d[y].values[ib]!=0]
        if cand.startswith('C3'): m=HR(max_iter=150).fit(d.iloc[ia][X],d[y].iloc[ia])
        else: m=LinearRegression().fit(d.iloc[ea][X],d[y].iloc[ea])
        err.append(np.abs(d[y].values[eb]-m.predict(d.iloc[eb][X])))
    return np.concatenate(err).mean()
pick={}; print('ROUTING RULE: simplest candidate within 10% of the lowest development-CV MAE (excursion layers, dev prints only)')
for c in Y:
    sc={k:cv_mae(c,k) for k in CANDS}; best=min(sc.values())
    pick[c]=[k for k in CANDS if sc[k]<=1.10*best][0]
    print(f'  {c:8s}',' | '.join(f'{k[:2]} {v:.3f}' for k,v in sc.items()),'->',pick[c])
Prule={}
for c in Y:
    y=Y[c]; r=res[c]; k=pick[c]
    if k.startswith('C3'): Prule[c]=gbm(raw+[r],proper,y)
    elif k.startswith('C2'): Prule[c]=lin([r+'_n1'],proper&(d[y]!=0),y)
    else: Prule[c]=lin([r],proper&(d[y]!=0),y)
P['PARD-rule (dev-only routing)']=Prule
G2={};PG2={}
for m,pc in P.items():
    for c,pr in pc.items(): PG2[(m,c)],G2[(m,c)]=gate_pred(pr,c)
Y_=np.array([d[Y[c]].values for c in Y]); 
tolA=TOL
# alt tolerance independent of excursions: 3 sigma of the plan-referenced residual on fully nominal development layers (layer_id>2)
nomdev=tr&(d.any_abnormal==0)&(d.layer_id>2)
tolB={c:3*d.loc[nomdev,res[c]].std() for c in Y}
print('tolerance A (hand): ',{c:round(v,3) for c,v in tolA.items()}); print('tolerance B (3 sigma of nominal dev-layer residual):',{c:round(v,3) for c,v in tolB.items()})
tm=te.values; rid=d.run_id.values
meths=['B1 same-layer residual (untuned)','B2 tuned per-channel residual','B3 learned, plan-aware (same layer)','PARD','PARD-rule (dev-only routing)']
for nm,tol in [('A hand tolerances',tolA),('B nominal-variability tolerances',tolB)]:
    Strue=np.max([np.abs(d[Y[c]].values)/tol[c] for c in Y],axis=0); lab=Strue>1
    S={m:score({c:PG2[(m,c)] for c in Y},tol) for m in meths}
    print(f'\n== tolerance {nm}: out-of-tol layers {lab[tm].sum()}/{tm.sum()}')
    for m in meths: print(f'  {m:40s} AUC {roc_auc_score(lab[tm],S[m][tm]):.3f}  F1 {f1_score(lab[tm],S[m][tm]>1):.3f}')
    # paired run-clustered bootstrap of differences
    ur=np.unique(rid[tm]); ix={r:np.where(rid==r)[0] for r in ur}; rng=np.random.default_rng(0); B=1000
    pairs=[('PARD','B3 learned, plan-aware (same layer)'),('PARD','B2 tuned per-channel residual'),('PARD','B1 same-layer residual (untuned)'),('PARD-rule (dev-only routing)','B3 learned, plan-aware (same layer)'),('PARD-rule (dev-only routing)','PARD')]
    D={p:{'AUC':[],'F1':[]} for p in pairs}
    for _ in range(B):
        ii=np.concatenate([ix[r] for r in rng.choice(ur,len(ur))]); 
        if lab[ii].min()==lab[ii].max(): continue
        for a,b in pairs:
            D[(a,b)]['AUC'].append(roc_auc_score(lab[ii],S[a][ii])-roc_auc_score(lab[ii],S[b][ii]))
            D[(a,b)]['F1'].append(f1_score(lab[ii],S[a][ii]>1)-f1_score(lab[ii],S[b][ii]>1))
    print('  paired run-clustered bootstrap of differences (1000 resamples):')
    for (a,b),v in D.items():
        print(f'   {a[:28]:28s} minus {b[:28]:28s} dAUC {np.mean(v["AUC"]):+.3f} [{np.percentile(v["AUC"],2.5):+.3f},{np.percentile(v["AUC"],97.5):+.3f}]  dF1 {np.mean(v["F1"]):+.3f} [{np.percentile(v["F1"],2.5):+.3f},{np.percentile(v["F1"],97.5):+.3f}]')

print('\nONSET detection: strict (flag on onset layer) vs allowing <=1 layer latency (flag on onset or next layer)')
for m in ['B1 same-layer residual (untuned)','B2 tuned per-channel residual','B3 learned, plan-aware (same layer)','PARD']:
    out=[]
    for c in Y:
        y=d[Y[c]].values; g=G2[(m,c)]; s=[];l=[]
        for r in np.unique(rid[tm]):
            ixr=np.where((rid==r)&tm)[0]; yy=y[ixr]; k=np.where(yy!=0)[0]
            if len(k)==0: continue
            sp=np.where((np.diff(k)!=1)|(np.sign(yy[k][1:])!=np.sign(yy[k][:-1])))[0]+1
            for blk in np.split(k,sp):
                a=ixr[blk[0]]; s.append(g[a]); l.append(g[a] or (a+1<len(g) and g[a+1]))
        out.append(f'{c} {np.mean(s):.2f}/{np.mean(l):.2f}')
    print(f'  {m:40s}',' | '.join(out))
