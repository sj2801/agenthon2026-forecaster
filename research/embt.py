"""Monday-as-of, 1-4 business-day backtest for the new Final series: EM FX (ECB crosses per USD),
UST bills, plus G10 FX for reference.  Mirrors the short-horizon practice design (as-of Monday,
targets Tuesday-Friday).  Engine = claude_final; scores = official M0-normalised composite."""
import pathlib
import pickle
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, '.')
import t2agent.engine as E  # noqa: E402  (claude_final engine first)
sys.path.append('research')
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from m0lib import m0_divisors, m0_draws, m0_mean_cov, normalized_composite  # noqa: E402
from t2agent.data import Unit  # noqa: E402
from t2agent.engine import Params, build_backbone, sample  # noqa: E402
from t2agent.events import FOMC_DECISIONS  # noqa: E402
from t2agent.params import TUNED  # noqa: E402

U = pathlib.Path('../track2-forecasting-public/units')
SRC = {'em': U / 't2-F2-fx-krw-015-ps1/em_fx_ecb_daily.parquet',
       'rates': U / 't2-F2-curve-ust6m-004-ps1/rates_daily.parquet',
       'g10': U / 't2-F3-election-2024-joint/g10_fx_daily.parquet'}
TYPES = [
    ('mxn', 'em', ['MXN'], [1, 2, 3, 4]), ('brl', 'em', ['BRL'], [1, 2, 3, 4]), ('zar', 'em', ['ZAR'], [1, 2, 3, 4]),
    ('krw', 'em', ['KRW'], [1, 2, 3, 4]), ('inr', 'em', ['INR'], [1, 2, 3, 4]),
    ('em3a', 'em', ['MXN', 'BRL', 'ZAR'], [2, 4]), ('em3b', 'em', ['KRW', 'INR', 'ZAR'], [4]),
    ('ust3m', 'rates', ['UST_3M'], [1, 2, 3, 4]), ('ust6m', 'rates', ['UST_6M'], [1, 2, 3, 4]),
    ('ust1y', 'rates', ['UST_1Y'], [1, 2, 3, 4]), ('bills4', 'rates', ['UST_3M', 'UST_6M', 'UST_1Y', 'UST_2Y'], [1, 2, 3, 4]),
    ('eur', 'g10', ['EUR'], [1, 2, 3, 4]), ('jpy', 'g10', ['JPY'], [1, 2, 3, 4]), ('g3', 'g10', ['EUR', 'JPY', 'CHF'], [3]),
]
P = {}
for k, p in SRC.items():
    df = pd.read_parquet(p); df['date'] = df['date'].astype(str).str[:10]
    P[k] = df.pivot_table(index='date', columns='asset', values='value').sort_index()
CACHE = pathlib.Path('.cache/embt_cards.pkl')
FD = np.array(FOMC_DECISIONS, dtype='datetime64[D]')


def unit_of(c):
    u = Unit(unit_id=f"em-{c['name']}-{c['origin']}", asof=c['origin'], card={}, spec=None, card_path=pathlib.Path('.'),
             assets=c['assets'], horizons=c['hs'], target_type='level', target_frequency='daily', family='', panels={})
    u.steps = np.tile(np.array(c['hs']), (len(c['assets']), 1))
    return u


def hist_of(c):
    W = P[c['panel']][c['assets']].dropna()
    W = W[W.index <= c['origin']].iloc[-2600:]
    return {a: W[a] for a in c['assets']}


def make_cards():
    cards = []
    for name, panel, assets, hs in TYPES:
        W = P[panel][assets].dropna()
        idx = list(W.index); dts = pd.to_datetime(pd.Series(idx))
        start = '2011-01-01' if panel == 'em' else '2003-01-01'
        for pos in range(len(idx) - 5):
            if idx[pos] < start or dts[pos].weekday() != 0:
                continue
            # the next four rows must be the same week's Tue-Fri (no holiday gaps)
            nxt = dts[pos + 1:pos + 5]
            if len(nxt) < 4 or (nxt.iloc[-1] - dts[pos]).days != 4:
                continue
            y = np.array([float(W[a].iloc[pos + h]) for a in assets for h in hs])
            lo, hi = np.datetime64(idx[pos], 'D'), np.datetime64(idx[pos + 4], 'D')
            cards.append(dict(name=name, panel=panel, assets=assets, hs=hs, origin=idx[pos], y=y,
                              fomc=bool(np.any((FD > lo) & (FD <= hi)))))
    return cards


def _prep(c):
    u = unit_of(c); h = hist_of(c)
    mean, C, L = m0_mean_cov(h, u.assets, u.steps, 'level')
    div = m0_divisors(mean, C)
    return dict(c, div=div, m0=normalized_composite(m0_draws(u.unit_id, mean, L, 500), c['y'], div)['composite'])


V = [('final', {}), ('no events', {'event_shrink': 0.0}), ('events 1.0', {'event_shrink': 1.0}),
     ('vm1.00', {'vol_mult': 1.0}), ('vm1.10', {'vol_mult': 1.10}), ('vm1.20', {'vol_mult': 1.20}),
     ('nu4', {'nu': 4.0}), ('nu8', {'nu': 8.0}), ('ewma10', {'ewma_hl': 10.0}), ('ewma40', {'ewma_hl': 40.0}),
     ('volhl30', {'vol_hl': 30.0}), ('lr750', {'lr_window': 750})]


def _eval(c):
    h = hist_of(c); u = unit_of(c); out = []
    for name, over in V:
        p = Params(**{**TUNED, **over})
        X = sample(build_backbone(u, p, h), p, 2000, 12345, None)
        out.append((c['name'], c['panel'], c['origin'], c['fomc'], name, normalized_composite(X, c['y'], c['div'])['composite'], c['m0']))
    return out


if __name__ == '__main__':
    t0 = time.time()
    if CACHE.exists():
        cards = pickle.load(open(CACHE, 'rb'))
    else:
        with Pool(2) as pool:
            cards = pool.map(_prep, make_cards(), chunksize=40)
        pickle.dump(cards, open(CACHE, 'wb'))
    print(len(cards), 'cards', pd.Series([c['panel'] for c in cards]).value_counts().to_dict(), f'{time.time()-t0:.0f}s', flush=True)
    with Pool(2) as pool:
        res = pool.map(_eval, cards, chunksize=30)
    df = pd.DataFrame([r for rr in res for r in rr], columns=['name', 'panel', 'origin', 'fomc', 'variant', 'score', 'm0'])
    df.to_pickle('.cache/embt.pkl')
    pd.set_option('display.width', 250)
    t = df.pivot_table(index='variant', columns=['panel', 'fomc'], values='score', aggfunc='mean').loc[[v[0] for v in V]]
    m0 = df[df.variant == 'final'].pivot_table(index='variant', columns=['panel', 'fomc'], values='m0', aggfunc='mean').rename(index={'final': 'M0'})
    print(pd.concat([m0, t]).round(4).to_string())
    print(df[df.variant == 'final'].groupby(['panel', 'fomc']).size().to_string())
    print(f'{time.time()-t0:.0f}s')
