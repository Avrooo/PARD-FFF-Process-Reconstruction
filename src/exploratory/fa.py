import warnings,sys; warnings.filterwarnings('ignore')
src=open('i.py').read().split("cfg={'thermal'")[0]
exec(src)
d['rT_n1']=d.groupby('run_id').rT.shift(-1); d['rT_n1']=d['rT_n1'].fillna(d.rT)
y='temp_cmd_deviation_C'
fn=mk('lin',['rT_n1'],proper,y); R=evaluate(fn,y)   # delay FIXED at exactly one layer
print('FIXED one-layer: MAE exc %.2f recall %.2f FAR %.3f cover %.2f bias %.2f width %.2f'%(R['MAEexc'],R['recall'],R['FAR'],R['cover'],R['bias_big'],R['width']))
t=d.loc[te].copy(); t['pred']=R['pred']; t['gate']=R['g']; t['exc']=R['x']
# layers since the last true excursion layer (within run), and until the next
t['last_exc']=t.layer_id.where(t.exc); t['last_exc']=t.groupby('run_id').last_exc.ffill()
t['since_exc']=t.layer_id-t.last_exc
nom=t[~t.exc]
print('nominal test layers',len(nom),' false alarms',int(nom.gate.sum()))
print(nom.groupby(nom.since_exc.fillna(-1).clip(upper=5)).agg(n=('gate','size'),FAR=('gate','mean')).round(3).rename_axis('layers since last excursion (-1=none yet)'))
print('by layer position:'); print(nom.groupby(pd.cut(nom.layer_id,[-1,0,2,10,40,70])).agg(n=('gate','size'),FAR=('gate','mean')).round(3))
fa=nom[nom.gate]; print('false alarms with since_exc<=2:',round(((fa.since_exc<=2)&fa.since_exc.notna()).mean(),2),' layer 0-2:',round((fa.layer_id<=2).mean(),2))
# true-abnormal-by-other-channel layers? (nominal for thermal but other faults active)
other=nom[(nom.any_abnormal==1)]; clean=nom[nom.any_abnormal==0]
print('FAR on nominal-thermal layers with other faults active %.3f (n=%d) vs fully clean layers %.3f (n=%d)'%(other.gate.mean(),len(other),clean.gate.mean(),len(clean)))
# FAR if we also exclude layer<=1 and 1-layer post-excursion settling (pre-specified blanking)
b=nom[(nom.layer_id>1)&~(nom.since_exc<=1)]; print('FAR after blanking layers 0-1 and 1 layer post-excursion: %.3f (n=%d)'%(b.gate.mean(),len(b)))
