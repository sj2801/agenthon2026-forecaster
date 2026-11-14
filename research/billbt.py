import sys, pickle, time
sys.path.insert(0, '.')
import t2agent.engine as E
sys.path.append('research')
import numpy as np, pandas as pd
from multiprocessing import Pool
import embt
from embt import hist_of, unit_of
from t2agent.engine import Params, build_backbone, sample
from t2agent.params import TUNED
from m0lib import normalized_composite
V = [('final', {}), ('vm1.0 nu4', {'vol_mult': 1.0, 'nu': 4.0}), ('vm1.0 nu4 noev', {'vol_mult': 1.0, 'nu': 4.0, 'event_shrink': 0.0}),
     ('vm1.0 nu5 noev', {'vol_mult': 1.0, 'nu': 5.0, 'event_shrink': 0.0}), ('vm0.95 nu4', {'vol_mult': 0.95, 'nu': 4.0}),
     ('vm1.0 nu4 ewma10', {'vol_mult': 1.0, 'nu': 4.0, 'ewma_hl': 10.0})]
def ev(c):
    h = hist_of(c); u = unit_of(c); out = []
    for n, o in V:
        p = Params(**{**TUNED, **o}); X = sample(build_backbone(u, p, h), p, 2000, 12345, None)
        out.append((c['name'], c['origin'], c['fomc'], n, normalized_composite(X, c['y'], c['div'])['composite'], c['m0']))
    return out
if __name__ == '__main__':
    cards = [c for c in pickle.load(open('.cache/embt_cards.pkl', 'rb')) if c['panel'] == 'rates']
    with Pool(2) as pool: res = pool.map(ev, cards, chunksize=30)
    df = pd.DataFrame([r for rr in res for r in rr], columns=['name', 'origin', 'fomc', 'variant', 'score', 'm0'])
    df['period'] = np.where(df.origin < '2013-01-01', '2003-12', '2013-22')
    df['bills_only'] = df.name != 'bills4'
    pd.set_option('display.width', 220)
    t = df.pivot_table(index='variant', columns=['period', 'fomc'], values='score', aggfunc='mean').loc[[v[0] for v in V]]
    t['all'] = df.groupby('variant').score.mean(); t['bills-only cards'] = df[df.bills_only].groupby('variant').score.mean(); t['bills4 card'] = df[~df.bills_only].groupby('variant').score.mean()
    print(t.round(4).to_string()); print('M0 all %.4f' % df[df.variant == 'final'].m0.mean())
