import time, pandas as pd
from finalbt import prepare, evaluate, table
t0=time.time(); cards=prepare()
M=[0.95,1.0,1.05,1.1]
V=[('nu=5 (base)',{},'',M),('nu=6',{'nu':6.0},'',M),('nu=7',{'nu':7.0},'',M),('nu=10',{'nu':10.0},'',M),('nu=7 ewma15',{'nu':7.0,'ewma_hl':15.0},'',M)]
df=evaluate(cards,V); df.to_pickle('.cache/fb3.pkl')
pd.set_option('display.width',200)
t=table(df); b=t.loc[('nu=5 (base)',1.0)]
print((t-b).round(4).to_string()); print(t.round(4).to_string()); print('%.0fs'%(time.time()-t0))
# component view at best settings
print(df.groupby(['variant','mult'])[['marg','joint','tail']].mean().round(4).to_string())
