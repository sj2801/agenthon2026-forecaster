"""Final-like backtest: UST yields + G10 FX, horizons 2-12 business days, unbiased origins.

The ranked Final uses fresh cards built before the outcome is known, so pseudo-cards are cut at
evenly spaced origins (no hindsight selection).  Every card is scored on the official
M0-normalized composite; outcomes are read only for scoring.
"""
import pathlib
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, '.'); sys.path.insert(0, 'research')
from m0lib import m0_divisors, m0_draws, m0_mean_cov, normalized_composite  # noqa: E402
from t2agent.data import Unit  # noqa: E402
from t2agent.engine import Adjust, Params, build_backbone, sample, stress_adjust  # noqa: E402
from t2agent.params import TUNED  # noqa: E402

U = pathlib.Path('../track2-forecasting-public/units')
SRC = {'rates': U / 't2-F1-hawkish-cut-2024/rates_daily.parquet',
       'fx': U / 't2-F3-election-2024-joint/g10_fx_daily.parquet'}
CARD_TYPES = [
    ('ust2y', 'rates', ['UST_2Y'], [2, 5, 10]),
    ('ust5y', 'rates', ['UST_5Y'], [3, 7, 12]),
    ('ust10y', 'rates', ['UST_10Y'], [2, 5, 10]),
    ('ust30y_12', 'rates', ['UST_30Y'], [12]),
    ('ust10y_5', 'rates', ['UST_10Y'], [5]),
    ('curve4', 'rates', ['UST_2Y', 'UST_5Y', 'UST_10Y', 'UST_30Y'], [3, 7, 12]),
    ('curve2', 'rates', ['UST_2Y', 'UST_10Y'], [5, 10]),
    ('eur', 'fx', ['EUR'], [2, 5, 10]),
    ('jpy', 'fx', ['JPY'], [3, 7, 12]),
    ('gbp_12', 'fx', ['GBP'], [12]),
    ('aud', 'fx', ['AUD'], [2, 5, 10]),
    ('chf_5', 'fx', ['CHF'], [5]),
    ('cad', 'fx', ['CAD'], [3, 7, 12]),
    ('noksek', 'fx', ['NOK', 'SEK'], [5, 10]),
    ('fx4', 'fx', ['EUR', 'GBP', 'JPY', 'CHF'], [3, 7, 12]),
    ('fxcomm', 'fx', ['AUD', 'NZD', 'CAD', 'NOK'], [2, 5, 10]),
    ('g10', 'fx', ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD', 'NOK', 'SEK'], [5, 12]),
]
CACHE = pathlib.Path('.cache/finalbt_cards.pkl')
KEEP = 2600


def load_panels():
    P = {}
    for k, p in SRC.items():
        df = pd.read_parquet(p)
        df['date'] = df['date'].astype(str).str[:10]
        W = df.pivot_table(index='date', columns='asset', values='value').sort_index().dropna()
        P[k] = W
    return P


P = load_panels()


def unit_of(c, family=''):
    u = Unit(unit_id=f"fb-{c['name']}-{c['origin']}", asof=c['origin'], card={}, spec=None,
             card_path=pathlib.Path('.'), assets=c['assets'], horizons=c['hs'], target_type='level',
             target_frequency='daily', family=family, panels={})
    u.steps = np.tile(np.array(c['hs']), (len(c['assets']), 1))
    return u


def hist_of(c):
    W = P[c['panel']]
    lo = max(0, c['pos'] - KEEP)
    return {a: W[a].iloc[lo:c['pos'] + 1] for a in c['assets']}


def make_cards(every=15, start='2003-01-01'):
    cards = []
    for name, panel, assets, hs in CARD_TYPES:
        W = P[panel]
        idx = list(W.index)
        p0 = next(i for i, d in enumerate(idx) if d >= start)
        for pos in range(p0, len(idx) - max(hs) - 1, every):
            y = np.array([float(W[a].iloc[pos + h]) for a in assets for h in hs])
            cards.append(dict(name=name, panel=panel, assets=assets, hs=hs, pos=pos, origin=idx[pos], y=y))
    return cards


def _prep(c):
    u = unit_of(c)
    h = hist_of(c)
    mean, C, L = m0_mean_cov(h, u.assets, u.steps, 'level')
    div = m0_divisors(mean, C)
    m0 = normalized_composite(m0_draws(u.unit_id, mean, L, 500), c['y'], div)['composite']
    return dict(c, div=div, m0=m0)


def prepare(every=15):
    if CACHE.exists():
        return pickle.load(open(CACHE, 'rb'))
    cards = make_cards(every)
    with Pool(2) as pool:
        cards = pool.map(_prep, cards, chunksize=50)
    pickle.dump(cards, open(CACHE, 'wb'))
    return cards


# ------------------------------------------------------------------------------- evaluation
N = 2000


def _eval(args):
    c, variants = args
    h = hist_of(c)
    out = []
    for vname, over, family, mults in variants:
        p = Params(**{**TUNED, **over})
        u = unit_of(c, family)
        bb = build_backbone(u, p, h)
        adj = Adjust(stress=stress_adjust(u, p)) if family else None
        X = sample(bb, p, N, 12345, adj)
        for m in mults:
            Xm = X if m == 1.0 else bb.mean[None, :] + m * (X - bb.mean[None, :])
            r = normalized_composite(Xm, c['y'], c['div'])
            out.append((c['name'], c['origin'], vname, m, r['composite'], r['marginal'], r['joint'], r['tail'], c['m0']))
    return out


def evaluate(cards, variants):
    with Pool(2) as pool:
        res = pool.map(_eval, [(c, variants) for c in cards], chunksize=40)
    rows = [r for rr in res for r in rr]
    df = pd.DataFrame(rows, columns=['name', 'origin', 'variant', 'mult', 'score', 'marg', 'joint', 'tail', 'm0'])
    df['period'] = np.where(df.origin < '2014-01-01', '2003-13', '2014-24')
    df['recent'] = df.origin >= '2021-01-01'
    return df


def table(df):
    g = df.groupby(['variant', 'mult'])
    t = pd.DataFrame({
        'all': g.score.mean(),
        '2003-13': df[df.period == '2003-13'].groupby(['variant', 'mult']).score.mean(),
        '2014-24': df[df.period == '2014-24'].groupby(['variant', 'mult']).score.mean(),
        '2021-24': df[df.recent].groupby(['variant', 'mult']).score.mean(),
        'rates': df[df.name.str.startswith(('ust', 'curve'))].groupby(['variant', 'mult']).score.mean(),
        'fx': df[~df.name.str.startswith(('ust', 'curve'))].groupby(['variant', 'mult']).score.mean(),
        'win_vs_M0': g.apply(lambda x: (x.score < x.m0).mean()),
    })
    return t


if __name__ == '__main__':
    t0 = time.time()
    cards = prepare()
    print(f'{len(cards)} cards ready in {time.time()-t0:.0f}s', flush=True)
    m0 = pd.DataFrame([(c['name'], c['origin'], c['m0']) for c in cards], columns=['name', 'origin', 'm0'])
    print('M0 mean: all %.4f | 2003-13 %.4f | 2014-24 %.4f | 2021-24 %.4f' % (
        m0.m0.mean(), m0[m0.origin < '2014'].m0.mean(), m0[m0.origin >= '2014'].m0.mean(), m0[m0.origin >= '2021'].m0.mean()))
