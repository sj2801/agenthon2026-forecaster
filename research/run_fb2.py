import time, pandas as pd
from finalbt import prepare, evaluate, table
t0=time.time(); cards=prepare()
M=[1.0,1.05]
V=[('base',{},'',M)]
for v in [5,10,15,30]: V.append((f'ewma_hl={v}',{'ewma_hl':float(v)},'',M))
for v in [20,50,400]: V.append((f'vol_hl={v}',{'vol_hl':float(v)},'',M))
for v in [4.0,7.0,10.0]: V.append((f'nu={v}',{'nu':v},'',M))
for v in [750,1250]: V.append((f'lr_window={v}',{'lr_window':v},'',M))
for v in [30,120]: V.append((f'corr_hl={v}',{'corr_hl':float(v)},'',M))
for v in [0.0,0.5]: V.append((f'corr_shrink={v}',{'corr_shrink':v},'',M))
df=evaluate(cards,V); df.to_pickle('.cache/fb2.pkl')
pd.set_option('display.width',200)
t=table(df); b=t.loc[('base',1.0)]
print(t.round(4).to_string()); print('\nchange vs base@1.0 (negative = better)'); print((t-b).drop(columns='win_vs_M0').round(4).to_string()); print('%.0fs'%(time.time()-t0))
