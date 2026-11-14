"""Out-of-sample test of the adaptive FOMC event model in claude_final's engine (1-4 bd cards)."""
import sys, pickle, time
sys.path.insert(0, '.')
import t2agent.engine as E  # claude_final engine first
sys.path.append('research')
import numpy as np, pandas as pd
from multiprocessing import Pool
from t2agent.engine import Params, build_backbone, sample
from t2agent.params import TUNED
import finalbt
from m0lib import normalized_composite
from shortbt import fomc_flag
cards = [c for c in pickle.load(open('.cache/shortbt_cards.pkl', 'rb')) if c['origin'] >= '2017-07-01']
V = [('vm1.05 no events', {'event_shrink': 0.0}, [1.0]),
     ('vm1.00 no events', {'event_shrink': 0.0, 'vol_mult': 1.0}, [1.0]),
     ('vm1.00 events 0.5', {'event_shrink': 0.5, 'vol_mult': 1.0}, [1.0]),
     ('vm1.00 events 0.75', {'event_shrink': 0.75, 'vol_mult': 1.0}, [1.0]),
     ('vm1.00 events 1.0', {'event_shrink': 1.0, 'vol_mult': 1.0}, [1.0]),
     ('vm1.05 events 0.75', {'event_shrink': 0.75}, [1.0])]
def ev(c):
    h = finalbt.hist_of(c); out = []
    for name, over, mults in V:
        p = Params(**{**TUNED, **over})
        bb = build_backbone(finalbt.unit_of(c), p, h)
        X = sample(bb, p, 2000, 12345, None)
        out.append((c['name'], c['origin'], name, normalized_composite(X, c['y'], c['div'])['composite'], c['m0'], fomc_flag(c)))
    return out
if __name__ == '__main__':
    t0 = time.time()
    with Pool(2) as pool: res = pool.map(ev, cards, chunksize=40)
    df = pd.DataFrame([r for rr in res for r in rr], columns=['name', 'origin', 'variant', 'score', 'm0', 'fomc'])
    df['cls'] = np.where(df.name.str.startswith(('ust', 'curve')), 'rates', 'fx')
    df.to_pickle('.cache/shortbt2.pkl')
    g = lambda d: d.groupby('variant').score.mean()
    t = pd.DataFrame({'all': g(df), 'FOMC in window': g(df[df.fomc]), 'no FOMC': g(df[~df.fomc]),
                      'rates FOMC': g(df[df.fomc & (df.cls == 'rates')]), 'fx FOMC': g(df[df.fomc & (df.cls == 'fx')]),
                      '2021-24 FOMC': g(df[df.fomc & (df.origin >= '2021')])})
    pd.set_option('display.width', 200); print(t.round(4).loc[[v[0] for v in V]].to_string())
    m = df.drop_duplicates(['name', 'origin'])
    print('M0: all %.4f | FOMC %.4f | no FOMC %.4f ; cards %d (FOMC %d)' % (m.m0.mean(), m[m.fomc].m0.mean(), m[~m.fomc].m0.mean(), len(m), m.fomc.sum()))
    print('%.0fs' % (time.time() - t0))
