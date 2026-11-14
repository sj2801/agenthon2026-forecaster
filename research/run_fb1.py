import time, pandas as pd
from finalbt import prepare, evaluate, table
t0=time.time(); cards=prepare()
V=[('engine',{},'',[0.85,0.9,0.95,1.0,1.05,1.1,1.15,1.25,1.5,1.75,2.5]),
   ('asis_F4',{},'T2-F4',[1.0])]
df=evaluate(cards,V); df.to_pickle('.cache/fb1.pkl')
pd.set_option('display.width',200)
print(table(df).round(4).to_string()); print('%.0fs'%(time.time()-t0))
