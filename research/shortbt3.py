import sys, pickle
sys.path.insert(0, '.')
import t2agent.engine as E
sys.path.append('research')
import numpy as np, pandas as pd
from multiprocessing import Pool
from t2agent.engine import Params, build_backbone, sample
from t2agent.params import TUNED
import finalbt
from m0lib import normalized_composite
from shortbt import fomc_flag
cards = [c for c in pickle.load(open('.cache/shortbt_cards.pkl', 'rb')) if c['origin'] >= '2017-07-01']
cards = [c for c in cards if fomc_flag(c)] + [c for c in cards if not fomc_flag(c)][::4]
V = [('vm1.05 events 0.75', {'event_shrink': 0.75}), ('vm1.05 events 1.0', {'event_shrink': 1.0}),
     ('vm1.10 events 0.75', {'event_shrink': 0.75, 'vol_mult': 1.10}), ('vm1.10 events 1.0', {'event_shrink': 1.0, 'vol_mult': 1.10}),
     ('vm1.05 events 0.75 nu5', {'event_shrink': 0.75, 'nu': 5.0}), ('vm1.05 events 0.75 nu8', {'event_shrink': 0.75, 'nu': 8.0})]
def ev(c):
    h = finalbt.hist_of(c); out = []
    for name, over in V:
        p = Params(**{**TUNED, **over}); bb = build_backbone(finalbt.unit_of(c), p, h)
        out.append((c['name'], c['origin'], name, normalized_composite(sample(bb, p, 2000, 12345, None), c['y'], c['div'])['composite'], fomc_flag(c)))
    return out
if __name__ == '__main__':
    with Pool(2) as pool: res = pool.map(ev, cards, chunksize=20)
    df = pd.DataFrame([r for rr in res for r in rr], columns=['name', 'origin', 'variant', 'score', 'fomc'])
    df['cls'] = np.where(df.name.str.startswith(('ust', 'curve')), 'rates', 'fx')
    t = pd.DataFrame({'FOMC': df[df.fomc].groupby('variant').score.mean(), 'no FOMC (1/4 sample)': df[~df.fomc].groupby('variant').score.mean(),
                      'rates FOMC': df[df.fomc & (df.cls == 'rates')].groupby('variant').score.mean(), 'fx FOMC': df[df.fomc & (df.cls == 'fx')].groupby('variant').score.mean()})
    print(t.round(4).loc[[v[0] for v in V]].to_string())
