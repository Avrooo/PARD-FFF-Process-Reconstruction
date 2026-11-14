import json,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
(ROOT / 'gen').mkdir(exist_ok=True)
(ROOT / 'figs').mkdir(exist_ok=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':7.5,'axes.linewidth':0.6,'axes.labelsize':7.5,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.8,'figure.dpi':200,'savefig.bbox':'tight','axes.spines.top':False,'axes.spines.right':False})
R=json.load(open('gen/results.json')); ST=pd.read_pickle('gen/store.pkl'); d=ST['d']; s1=ST['store1']; Y=ST['Y']; TOL=ST['TOL']
SENS=pd.DataFrame(R['sens']); CH=pd.DataFrame(R['chan']); AB=pd.DataFrame(R['abl'])
NM={'B1':'Bone','B2':'Btwo','B3':'Bthree','PARD':'Pard'}
M=[]
def mac(n,v): M.append('\\newcommand{\\%s}{%s}'%(n,v))
def f(x,k=3): return f'{x:.{k}f}'
# --- counts
for k,n in [('layers_total','layersTotal'),('layers_degenerate','layersDegenerate'),('layers_used','layersUsed'),('layers_dev','layersDev'),('layers_test','layersTest'),('prints_dev','printsDev'),('prints_test','printsTest'),('n_raw_features','nRaw')]: mac(n,R[k])
mac('samplingHz',f(R['sampling_hz_median'],2)); mac('layerDur',f(R['layer_duration_median_s'],1))
pc=R['print_counts']; mac('nNormal',pc['Normal']); mac('nBed',pc['Bed adhesion']); mac('nCool',pc['Cooling']); mac('nFlow',pc['Flow']); mac('nSpeed',pc['Speed']); mac('nThermal',pc['Thermal']); mac('nMixed',pc['Mixed: Thermal+Cooling+Speed+Flow'])
mac('nSingle',pc['Cooling']+pc['Flow']+pc['Speed']+pc['Thermal'])
# --- headline (tolerance A)
A=SENS[SENS.tol=='A'].set_index('method')
for m,n in NM.items():
    mac('auc'+n,f(A.loc[m,'AUC'])); mac('aucSd'+n,f(A.loc[m,'AUCsd'])); mac('fone'+n,f(A.loc[m,'F1'])); mac('foneSd'+n,f(A.loc[m,'F1sd']))
    mac('prec'+n,f(A.loc[m,'prec'])); mac('rec'+n,f(A.loc[m,'rec'])); mac('smae'+n,f(A.loc[m,'Smae'],2)); mac('smaeSd'+n,f(A.loc[m,'Smaesd'],2))
mac('nOot',int(A.loc['PARD','n_oot']))
for k,n in [('PARD-B3','Bthree'),('PARD-B2','Btwo'),('PARD-B1','Bone')]:
    for met,mn in [('AUC','dAuc'),('F1','dFone'),('Smae','dSmae')]:
        v=R['paired'][k][met]; mac(mn+n,f'{v[0]:+.3f}'.replace('+','$+$').replace('-','$-$')); mac(mn+n+'Lo',f'{v[1]:+.3f}'.replace('+','$+$').replace('-','$-$')); mac(mn+n+'Hi',f'{v[2]:+.3f}'.replace('+','$+$').replace('-','$-$'))
# --- thermal ablation and per-channel
ab=AB.set_index(['chan','variant'])
for v,n in [('delay0','DelayZero'),('delay1','DelayOne'),('delay2','DelayTwo'),('learned','Learned')]:
    mac('th'+n,f(ab.loc[('thermal',v),'MAE'],2)); mac('th'+n+'Sd',f(ab.loc[('thermal',v),'MAEsd'],2))
d0,d1=ab.loc[('thermal','delay0'),'MAE'],ab.loc[('thermal','delay1'),'MAE']; mac('thRedPct',f(100*(1-d1/d0),0))
b2=CH[(CH.chan=='thermal')&(CH.method=='B2')].MAE.iloc[0]; mac('thBtwo',f(b2,2)); mac('thRedBtwoPct',f(100*(1-d1/b2),0))
cv=R['cv']; mac('cvThDrop',f(100*(1-cv['thermal']['C2']/cv['thermal']['C1']),0))
for c,cn in [('thermal','Th'),('cooling','Co'),('speed','Sp'),('flow','Fl')]:
    for k in ['C1','C2','C3']: mac(f'cv{cn}{k[0]}{["one","two","three"][int(k[1])-1]}',f(cv[c][k],3 if c=='cooling' else 2))
    for m,n in NM.items():
        r=CH[(CH.chan==c)&(CH.method==m)].iloc[0]
        mac(f'mae{cn}{n}',f(r.MAE,3 if c=='cooling' else 2)); mac(f'maeSd{cn}{n}',f(r.MAEsd,3 if c=='cooling' else 2))
        mac(f'rcl{cn}{n}',f(r.recall,2)); mac(f'far{cn}{n}',f(r.FAR,3))
        if r.cover==r.cover: mac(f'cov{cn}{n}',f(r.cover,2)); mac(f'covMin{cn}{n}',f(r.cover_min,2))
        else: mac(f'cov{cn}{n}','--'); mac(f'covMin{cn}{n}','--')
        mac(f'wid{cn}{n}',f(r.width,3 if c=='cooling' else 2) if r.width==r.width else '--')
# --- ranges
dr=R['dev_ranges']; mac('thDevMean',f(R['events_dev']['thermal']['mean_dev'],1)); 
mac('thMixMin',int(dr['thermal']['mix_min'])); mac('thMixMax',int(dr['thermal']['mix_max'])); mac('thDevMin',int(-dr['thermal']['dev_min']))
em=R['events_mixed']; mac('evLenMed',int(em['thermal']['len_med'])); mac('evLenMin',int(min(v['len_min'] for v in em.values()))); mac('evLenMax',int(max(v['len_max'] for v in em.values())))
od={r['since']:r['median'] for r in R['onset_dev']}; om={r['since']:r['median'] for r in R['onset_mixed']}
mac('onsetDevZero',f(od[0],2)); mac('onsetDevOne',f(od[1],2)); mac('onsetMixZero',f(om[0],2)); mac('onsetMixOne',f(om[1],2)); mac('onsetMixTwo',f(om[2],2))
for c,cn in [('thermal','Th'),('cooling','Co'),('speed','Sp'),('flow','Fl')]: mac('tol'+cn,TOL[c])
tb=R['tolB']; mac('tolBTh',f(tb['thermal'],1)); mac('tolBSp',f(tb['speed'],0)); mac('tolBFl',f(tb['flow'],0))
open('gen/results_macros.tex','w').write('\n'.join(M)+'\n')
# ================= tables
def tex_row(cells): return ' & '.join(cells)+' \\\\\n'
t='\\begin{tabular}{lcccc}\n\\toprule\n'+tex_row(['Channel','Same-layer linear','Delayed linear','Learned','Selected'])+'\\midrule\n'
sel={'C1':'Same-layer linear','C2':'Delayed linear','C3':'Learned'}
for c,lab,u in [('thermal','Thermal (\\si{\\celsius})',''),('cooling','Cooling (fan, 0--1)',''),('speed','Speed (\\%)',''),('flow','Flow (\\%)','')]:
    k=R['pick'][c]; v=R['cv'][c]; fm=(lambda x:f'{x:.3f}') if c=='cooling' else (lambda x:f'{x:.2f}')
    cells=[fm(v['C1']),fm(v['C2']),fm(v['C3'])]; cells[int(k[1])-1]='\\textbf{'+cells[int(k[1])-1]+'}'
    t+=tex_row([lab]+cells+[sel[k]])
t+='\\bottomrule\n\\end{tabular}\n'; open('gen/tab_routing.tex','w').write(t)
# table: sensitivity
t='\\begin{tabular}{llcccc}\n\\toprule\n'+tex_row(['Tolerance set','Metric','B1','B2','B3','PARD'])+'\\midrule\n'
for tn,lab in [('A','Principal (5 \\si{\\celsius}, 0.2, 10\\%, 10\\%)'),('Ax0.5','Principal $\\times0.5$'),('Ax2','Principal $\\times2$'),('B','Nominal $3\\sigma$')]:
    S_=SENS[SENS.tol==tn].set_index('method'); 
    for i,(met,col) in enumerate([('AUC','AUC'),('F1','F1')]):
        vals=[S_.loc[m,col] for m in ['B1','B2','B3','PARD']]; best=max(vals)
        cells=[('\\textbf{%.3f}'%v if v==best else '%.3f'%v) for v in vals]
        t+=tex_row([lab if i==0 else '',met]+cells)
t+='\\bottomrule\n\\end{tabular}\n'; open('gen/tab_sensitivity.tex','w').write(t)
# table: reliability for PARD
t='\\begin{tabular}{lccccc}\n\\toprule\n'+tex_row(['Channel','MAE','Recall','False alarm','Coverage','Interval width'])+'\\midrule\n'
for c,lab in [('thermal','Thermal (\\si{\\celsius})'),('cooling','Cooling (0--1)'),('speed','Speed (\\%)'),('flow','Flow (\\%)')]:
    r=CH[(CH.chan==c)&(CH.method=='PARD')].iloc[0]; fm=(lambda x:f'{x:.3f}') if c=='cooling' else (lambda x:f'{x:.2f}')
    t+=tex_row([lab,fm(r.MAE),f'{r.recall:.2f}',f'{r.FAR:.3f}',f'{r.cover:.2f}',fm(r.width)])
t+='\\bottomrule\n\\end{tabular}\n'; open('gen/tab_reliability.tex','w').write(t)
# ================= figures
C={'B1':'#9a9a9a','B2':'#c9a227','B3':'#4c78a8','PARD':'#c0392b'}; LBL={'B1':'B1 same-layer','B2':'B2 tuned residual','B3':'B3 learned, plan-aware','PARD':'PARD'}
# Fig 1 architecture
fig,ax=plt.subplots(figsize=(7.1,2.15)); ax.set_xlim(0,106); ax.set_ylim(0,30); ax.axis('off')
def box(x,y,w,h,txt,fc='#f2f2f2',ec='#333333',fs=6.8,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.25,rounding_size=0.8',fc=fc,ec=ec,lw=0.7)); ax.text(x+w/2,y+h/2,txt,ha='center',va='center',fontsize=fs,fontweight='bold' if bold else 'normal',linespacing=1.15)
def arr(x1,y1,x2,y2): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=7,lw=0.7,color='#333333'))
box(1,19.5,17,8,'G-code plan $u^{\\rm plan}$\n$T^{*},\\,f^{*},\\,v^{*},\\,e^{*}$',fc='#e8eef7')
box(1,3.5,17,8,'Firmware telemetry\n$y_{1:t+\\delta}$ (layer-level)',fc='#e8eef7')
box(24,11.5,17,9.5,'Plan-referenced\nresiduals $r_{c}$\n(thermal, fan, speed, extrusion)',fc='#fdf3e3')
box(46,11.5,17,9.5,'Response alignment\n$r_{c,\\,t+\\delta_c}$,\nselected by CV',fc='#fdf3e3')
box(68,19.5,17,8,'Channel estimator\n(selected on development\nprints)',fc='#fdf3e3',bold=False)
ax.text(76.5,16.2,'linear  |  delayed linear  |  learned',ha='center',fontsize=6.2,style='italic')
box(68,3.5,17,8,'Gate and interval\n$|\\hat d_c|>\\theta_c$',fc='#fdf3e3')
box(88,11.5,17,9.5,'Reconstructed\ndeviations $\\hat{d}_{c,t}$\nand score $S_t$',fc='#e4f2e4',bold=True)
ax.text(96.5,8.3,'$S_t=\\max_c\\,|\\hat{d}_{c,t}|/\\tau_c$',ha='center',fontsize=6.6)
arr(18.4,23.5,24,17.5); arr(18.4,7.5,24,14.5); arr(41.4,16.2,46,16.2); arr(63.4,18.5,68,22.5); arr(76.5,19.3,76.5,11.8); arr(85.4,23.5,88,19); arr(85.4,7.5,88,13.5)
ax.text(1,29.3,'Inputs',fontsize=7,fontweight='bold'); ax.text(24,29.3,'Reference and alignment',fontsize=7,fontweight='bold'); ax.text(68,29.3,'Reconstruction',fontsize=7,fontweight='bold'); ax.text(88,29.3,'Output',fontsize=7,fontweight='bold')
fig.savefig('figs/fig1_architecture.pdf'); plt.close(fig)
# Fig 2 representative trace
run=sorted(d.run_id[d.fault_type.str.startswith('Mixed')].unique())[0]; ix=np.where(d.run_id.values==run)[0]; lay=d.layer_id.values[ix]
fig,axs=plt.subplots(2,3,figsize=(7.1,3.5),sharex=True); axs=axs.ravel()
for a,(c,lab) in zip(axs[:4],[('thermal','Thermal deviation (\\si{}°C)'),('cooling','Fan deviation (0--1)'),('speed','Speed deviation (%)'),('flow','Flow deviation (%)')]):
    y=d[Y[c]].values[ix]; a.step(lay,y,where='mid',color='#222222',lw=1.0,label='Commanded deviation'); 
    a.plot(lay,s1['PG']['PARD',c][ix],'o-',ms=2.3,lw=0.7,color=C['PARD'],label='PARD (gated)')
    a.plot(lay,s1['PG']['B3',c][ix],'s--',ms=1.8,lw=0.6,color=C['B3'],label='B3 learned, plan-aware')
    a.set_ylabel(lab.replace('\\si{}°C','°C')); a.axhline(0,color='#bbbbbb',lw=0.4)
S=axs[4]; S.step(lay,s1['Strue'][ix],where='mid',color='#222222',lw=1.0); S.plot(lay,s1['S']['PARD'][ix],'o-',ms=2.3,lw=0.7,color=C['PARD']); S.plot(lay,s1['S']['B3'][ix],'s--',ms=1.8,lw=0.6,color=C['B3']); S.axhline(1,color='#2e7d32',lw=0.8,ls=':'); S.set_ylabel('Score $S_t$'); S.text(56,1.15,'tolerance ($S_t=1$)',fontsize=6,color='#2e7d32',va='bottom')
for a in axs[3:5]: a.set_xlabel('Layer index')
axs[2].set_xlabel('Layer index'); 
h,l=axs[0].get_legend_handles_labels(); axs[5].axis('off'); axs[5].legend(h,l,loc='center',frameon=False,fontsize=7); axs[5].text(0.5,0.1,'Print %s (mixed excursions)'%run.replace('_',' '),ha='center',transform=axs[5].transAxes,fontsize=7)
fig.tight_layout(); fig.savefig('figs/fig2_trace.pdf'); plt.close(fig)
# Fig 3 main comparison
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.1,2.35),gridspec_kw={'width_ratios':[1.15,1]})
x=np.arange(2); w=0.19
for i,m in enumerate(['B1','B2','B3','PARD']):
    r=A.loc[m]; a1.bar(x+(i-1.5)*w,[r.AUC,r.F1],w,yerr=[r.AUCsd,r.F1sd],color=C[m],edgecolor='black',lw=0.4,capsize=1.5,label=LBL[m],hatch=['','//','..',''][i])
    for xx,v in zip(x+(i-1.5)*w,[r.AUC,r.F1]): a1.text(xx,v+0.012,f'{v:.2f}',ha='center',fontsize=5.8)
a1.set_xticks(x); a1.set_xticklabels(['AUC','F1']); a1.set_ylim(0.5,1.17); a1.set_ylabel('Execution-fidelity discrimination'); a1.legend(loc='upper left',frameon=False,ncol=2,fontsize=6.0,columnspacing=0.8,handlelength=1.4); a1.set_title('(a) Five calibration splits, mean $\\pm$ s.d.',fontsize=7.5)
pr=R['paired']; labs=[('PARD$-$B3','PARD-B3'),('PARD$-$B2','PARD-B2'),('PARD$-$B1','PARD-B1')]
for j,(l,k) in enumerate(labs):
    for i,(met,off,mk) in enumerate([('AUC',-0.1,'o'),('F1',0.1,'s')]):
        v=pr[k][met]; a2.errorbar(v[0],j+off,xerr=[[v[0]-v[1]],[v[2]-v[0]]],fmt=mk,color=['#c0392b','#1f5f8b'][i],ms=3.5,capsize=2,lw=0.9,label=('$\\Delta$AUC' if (j==0 and met=='AUC') else ('$\\Delta$F1' if (j==0) else None)))
a2.axvline(0,color='#777777',lw=0.6); a2.set_yticks(range(3)); a2.set_yticklabels([l for l,_ in labs]); a2.set_xlabel('Paired difference (95% run-clustered bootstrap)'); a2.legend(frameon=False,loc='lower left',fontsize=6.5); a2.set_title('(b) Paired differences, seed-1 split',fontsize=7.5); a2.invert_yaxis()
fig.tight_layout(); fig.savefig('figs/fig3_comparison.pdf'); plt.close(fig)
# Fig 4 delay and step response
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.1,2.35),gridspec_kw={'width_ratios':[1,1.05]})
vars_=[('delay0','0 layers'),('delay1','1 layer'),('delay2','2 layers'),('learned','Learned')]; cols=['#9a9a9a','#c0392b','#e59a8f','#4c78a8']
for i,(v,l) in enumerate(vars_):
    m=ab.loc[('thermal',v),'MAE']; sd=ab.loc[('thermal',v),'MAEsd']; a1.bar(i,m,0.62,yerr=sd,color=cols[i],edgecolor='black',lw=0.4,capsize=2); a1.text(i,m+sd+0.4,f'{m:.1f}',ha='center',fontsize=6.5)
a1.set_xticks(range(4)); a1.set_xticklabels([l for _,l in vars_]); a1.set_ylabel('Thermal MAE (°C), mixed test'); a1.set_title('(a) Alignment delay ($\\delta_T$) ablation',fontsize=7.5)
for df,lab,mk,col in [(R['onset_dev'],'Development steps (cold)','s','#1f5f8b'),(R['onset_mixed'],'Mixed test steps (hot)','o','#c0392b')]:
    a2.plot([r['since'] for r in df],[r['median'] for r in df],mk+'-',ms=3.5,lw=0.9,color=col,label=lab)
a2.axhline(1,color='#777777',lw=0.6,ls=':'); a2.set_xlabel('Layers since excursion onset'); a2.set_ylabel('Measured / commanded deviation'); a2.legend(frameon=False,fontsize=6.5,loc='lower right'); a2.set_title('(b) Thermal step response',fontsize=7.5)
fig.tight_layout(); fig.savefig('figs/fig4_delay.pdf'); plt.close(fig)
print('assets ok'); print(open('gen/results_macros.tex').read()[:1500])
# ---- dataset table fragment
dr=R['dev_ranges']; pc=R['print_counts']
def sg(x,k=0):
    v=round(x,k); 
    if v==0: return '0'
    t=('%.*f'%(k,abs(v))); return ('$-$' if v<0 else '$+$')+t
def rg(c,pre): return sg(dr[c][pre+'_min'])+' to '+sg(dr[c][pre+'_max'])
t='\\begin{tabular}{lcc}\n\\toprule\n'+tex_row(['Print group','Prints','Role'])+'\\midrule\n'
t+=tex_row(['Nominal',str(pc['Normal']),'Development'])+tex_row(['Bed adhesion (no excursion in the four channels)',str(pc['Bed adhesion']),'Development'])
t+=tex_row(['Single excursion: thermal / cooling / speed / flow','%d / %d / %d / %d'%(pc['Thermal'],pc['Cooling'],pc['Speed'],pc['Flow']),'Development'])
t+=tex_row(['Mixed excursions (all four channels)',str(pc['Mixed: Thermal+Cooling+Speed+Flow']),'Evaluation'])
t+='\\midrule\n\\multicolumn{3}{l}{\\textit{Commanded deviation from plan, excursion layers (development $|$ mixed)}}\\\\\n'
t+=tex_row(['Thermal (\\si{\\celsius})','%s $|$ %s'%(rg('thermal','dev'),rg('thermal','mix')),''])
t+=tex_row(['Cooling (fan, 0--1)','%s to %s $|$ %s to %s'%(sg(dr['cooling']['dev_min'],2),sg(dr['cooling']['dev_max'],2),sg(dr['cooling']['mix_min'],2),sg(dr['cooling']['mix_max'],2)),''])
t+=tex_row(['Speed (\\%)','%s $|$ %s'%(rg('speed','dev'),rg('speed','mix')),''])
t+=tex_row(['Flow (\\%)','%s $|$ %s'%(rg('flow','dev'),rg('flow','mix')),''])
t+='\\bottomrule\n\\end{tabular}\n'; open('gen/tab_data.tex','w').write(t)
