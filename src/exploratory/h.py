import warnings; warnings.filterwarnings('ignore')
exec(open('g.py').read().split("alpha=.1")[0])
from sklearn.linear_model import LinearRegression
alpha=.1
print('\ncorr(plan-ref residual, true dev) on train excursion layers / mixed excursion layers')
for k,(r,y,f) in ch.items():
    a=d[tr&(d[y]!=0)]; b=d[te&(d[y]!=0)]
    print(f'{k:8s} {np.corrcoef(a[r],a[y])[0,1]:.3f} / {np.corrcoef(b[r],b[y])[0,1]:.3f}   slope(train OLS)={np.polyfit(a[r],a[y],1)[0]:.3f}')
print(f"\n{'chan':8s}{'method':13s}{'MAE exc':>8s}{'MAE exc>trainmax':>17s}{'bias>trainmax':>14s}{'cover exc':>10s}{'cover nom':>10s}{'width exc':>10s}")
for k,(r,y,f) in ch.items():
    ex_tr=proper&(d[y]!=0); 
    lin=LinearRegression().fit(d.loc[proper,[r]],d.loc[proper,y])                # all proper layers
    lin_e=LinearRegression().fit(d.loc[ex_tr,[r]],d.loc[ex_tr,y])                # fitted on excursion layers
    gb=HR(max_iter=150).fit(d.loc[proper,raw+[r]],d.loc[proper,y])
    trmax=d.loc[tr,y].abs().max()
    # gate threshold from nominal calibration layers: 99th pct |r|
    gate=np.quantile(np.abs(d.loc[cal&(d[y]==0),r]),.99)
    for nm,fn in [('plan-ref OLS',lambda m:lin.predict(d.loc[m,[r]])),('plan-ref exc-fit',lambda m:lin_e.predict(d.loc[m,[r]])),('GBM raw+res',lambda m:gb.predict(d.loc[m,raw+[r]]))]:
        pc=fn(cal); ec=cal&(d[y]!=0)
        pe=fn(ec); q_e=np.quantile(np.abs(d.loc[ec,y]-pe),1-alpha)*(1+1/ec.sum())   # Mondrian: excursion-layer calibration
        nc=cal&(d[y]==0); q_n=np.quantile(np.abs(d.loc[nc,y]-fn(nc)),1-alpha)
        yt=d.loc[te,y].values; pt=fn(te); x=yt!=0; e=np.abs(yt-pt)
        big=x&(np.abs(yt)>trmax)
        mb=e[big].mean() if big.any() else np.nan; bb=(pt[big]-yt[big]).mean() if big.any() else np.nan
        print(f'{k:8s}{nm:13s}{e[x].mean():8.2f}{mb:17.2f}{bb:14.2f}{np.mean(e[x]<=q_e):10.2f}{np.mean(e[~x]<=q_n):10.2f}{2*q_e:10.2f}   (n_beyond_train={big.sum()}, trainmax={trmax:.1f})')
