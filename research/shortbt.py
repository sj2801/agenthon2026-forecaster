"""Short-horizon (1-4 business days) Final-like backtest with an FOMC event-day variance model.

Event model: per-step variance v(k) is multiplied by w(k) / c, where w = r on the event step
(UST: the FOMC day itself; FX noon/ECB-style fixings: the next panel day), w = q on the FX
pre-statement day, 1 otherwise; c = 1 + (r-1)f + (q-1)f renormalises so that the average day keeps
the estimated variance (f = 8/252 FOMC days per year).
"""
import inspect
import pathlib
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, '.'); sys.path.insert(0, 'research')
import t2agent.engine as E  # noqa: E402
from events import FOMC  # noqa: E402
from finalbt import P, unit_of, hist_of  # noqa: E402
from m0lib import m0_divisors, m0_draws, m0_mean_cov, normalized_composite  # noqa: E402
from t2agent.engine import Adjust, Params, sample  # noqa: E402
from t2agent.params import TUNED  # noqa: E402

CARD_TYPES = [
    ('ust2y', 'rates', ['UST_2Y'], [1, 2, 3, 4]),
    ('ust10y', 'rates', ['UST_10Y'], [1, 2, 4]),
    ('ust5y_2', 'rates', ['UST_5Y'], [2]),
    ('ust30y', 'rates', ['UST_30Y'], [1, 3]),
    ('curve4', 'rates', ['UST_2Y', 'UST_5Y', 'UST_10Y', 'UST_30Y'], [1, 2, 3, 4]),
    ('curve2', 'rates', ['UST_2Y', 'UST_10Y'], [1, 4]),
    ('eur', 'fx', ['EUR'], [1, 2, 3, 4]),
    ('jpy', 'fx', ['JPY'], [1, 2, 4]),
    ('gbp_3', 'fx', ['GBP'], [3]),
    ('aud', 'fx', ['AUD'], [1, 4]),
    ('chf_2', 'fx', ['CHF'], [2]),
    ('fx4', 'fx', ['EUR', 'GBP', 'JPY', 'CHF'], [1, 2, 3, 4]),
    ('fxcomm', 'fx', ['AUD', 'NZD', 'CAD', 'NOK'], [1, 3]),
    ('g10', 'fx', ['EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD', 'NOK', 'SEK'], [2, 4]),
]
CACHE = pathlib.Path('.cache/shortbt_cards.pkl')
FDATES = np.array(sorted(pd.to_datetime(FOMC)), dtype='datetime64[D]')
R_UST = {'UST_2Y': 2.3, 'UST_5Y': 1.8, 'UST_7Y': 1.5, 'UST_10Y': 1.2, 'UST_20Y': 1.0, 'UST_30Y': 1.0}
R_FX, Q_FX, FREQ = 2.7, 0.65, 8 / 252


def make_cards(every=9, start='2003-01-01'):
    cards = []
    for name, panel, assets, hs in CARD_TYPES:
        W = P[panel]; idx = list(W.index)
        p0 = next(i for i, d in enumerate(idx) if d >= start)
        for pos in range(p0, len(idx) - max(hs) - 1, every):
            y = np.array([float(W[a].iloc[pos + h]) for a in assets for h in hs])
            step_dates = np.array(idx[pos + 1:pos + max(hs) + 1], dtype='datetime64[D]')
            prev_dates = np.array(idx[pos:pos + max(hs)], dtype='datetime64[D]')
            cards.append(dict(name=name, panel=panel, assets=assets, hs=hs, pos=pos, origin=idx[pos], y=y,
                              step_dates=step_dates, prev_dates=prev_dates))
    return cards


def _prep(c):
    u = unit_of(c); h = hist_of(c)
    mean, C, L = m0_mean_cov(h, u.assets, u.steps, 'level')
    div = m0_divisors(mean, C)
    m0 = normalized_composite(m0_draws(u.unit_id, mean, L, 500), c['y'], div)['composite']
    return dict(c, div=div, m0=m0)


def prepare():
    if CACHE.exists():
        return pickle.load(open(CACHE, 'rb'))
    cards = make_cards()
    with Pool(2) as pool:
        cards = pool.map(_prep, cards, chunksize=50)
    pickle.dump(cards, open(CACHE, 'wb'))
    return cards


def event_weights(c, shrink):
    """(assets x K) per-step variance weights for this card."""
    K = max(c['hs'])
    W = np.ones((len(c['assets']), K))
    if shrink <= 0:
        return W
    # step k covers (prev_date, step_date]; UST: FOMC day inside it; FX: the step whose fixing is the
    # first one after the FOMC day (the FOMC day itself is the quiet pre-statement fixing)
    for ai, a in enumerate(c['assets']):
        for k in range(K):
            lo, hi = c['prev_dates'][k], c['step_dates'][k]
            if c['panel'] == 'rates':
                r = 1 + shrink * (R_UST.get(a, 1.5) - 1)
                hit = np.any((FDATES > lo) & (FDATES <= hi))
                W[ai, k] = r / (1 + (r - 1) * FREQ) if hit else 1 / (1 + (r - 1) * FREQ)
            else:
                r = 1 + shrink * (R_FX - 1); q = 1 + shrink * (Q_FX - 1)
                after = np.any((FDATES >= lo) & (FDATES < hi))      # FOMC on/after previous fixing day, before this one
                on = np.any(FDATES == hi)
                c0 = 1 + (r - 1) * FREQ + (q - 1) * FREQ
                W[ai, k] = (r if after else (q if on else 1.0)) / c0
    return W


_src = inspect.getsource(E.build_backbone)
_old = "vpath = {a: s2_lr[a] + (s2_now[a] - s2_lr[a]) * phi**k for a in assets}"
assert _old in _src
_src = _src.replace(_old, _old + "\n    vpath = {a: vpath[a] * _EVW[ai_][: len(vpath[a])] for ai_, a in enumerate(assets)}")
_src = _src.replace('def build_backbone(', 'def build_backbone_ev(')
_ns = dict(vars(E)); _ns['_EVW'] = None
exec(compile(_src, 'bbev', 'exec'), _ns)
N = 2000


def _eval(args):
    c, variants = args
    h = hist_of(c); out = []
    for vname, over, shrink, mults in variants:
        p = Params(**{**TUNED, **over})
        _ns['_EVW'] = event_weights(c, shrink)
        bb = _ns['build_backbone_ev'](unit_of(c), p, h)
        X = sample(bb, p, N, 12345, None)
        for m in mults:
            Xm = X if m == 1.0 else bb.mean[None, :] + m * (X - bb.mean[None, :])
            r = normalized_composite(Xm, c['y'], c['div'])
            out.append((c['name'], c['origin'], vname, m, r['composite'], c['m0']))
    return out


def evaluate(cards, variants):
    with Pool(2) as pool:
        res = pool.map(_eval, [(c, variants) for c in cards], chunksize=40)
    return pd.DataFrame([r for rr in res for r in rr], columns=['name', 'origin', 'variant', 'mult', 'score', 'm0'])


def fomc_flag(c):
    lo, hi = c['prev_dates'][0], c['step_dates'][-1]
    if c['panel'] == 'rates':
        return bool(np.any((FDATES > lo) & (FDATES <= hi)))
    return bool(np.any((FDATES >= lo) & (FDATES < hi)))


if __name__ == '__main__':
    t0 = time.time()
    cards = prepare()
    print(f'{len(cards)} short-horizon cards in {time.time()-t0:.0f}s', flush=True)
    F = {'nu': 6.0, 'vol_mult': 1.0, 'fam_mult': {}, 'stress_tail': {}}
    V = [('final nu6', F, 0.0, [1.0, 1.05, 1.1]),
         ('final nu6 + events s0.5', F, 0.5, [1.0, 1.05]),
         ('final nu6 + events s0.75', F, 0.75, [1.0, 1.05]),
         ('final nu6 + events s1.0', F, 1.0, [1.0, 1.05]),
         ('nu5', {**F, 'nu': 5.0}, 0.0, [1.0]),
         ('nu8', {**F, 'nu': 8.0}, 0.0, [1.0]),
         ('ewma10', {**F, 'ewma_hl': 10.0}, 0.0, [1.0]),
         ('ewma30', {**F, 'ewma_hl': 30.0}, 0.0, [1.0])]
    df = evaluate(cards, V)
    flag = {(c['name'], c['origin']): fomc_flag(c) for c in cards}
    df['fomc'] = [flag[(n, o)] for n, o in zip(df.name, df.origin)]
    df['cls'] = np.where(df.name.str.startswith(('ust', 'curve')), 'rates', 'fx')
    from events import COMPLETE
    df['complete'] = pd.to_datetime(df.origin).dt.year.isin(COMPLETE)
    df.to_pickle('.cache/shortbt.pkl')
    pd.set_option('display.width', 220)
    g = df.groupby(['variant', 'mult'])
    t = pd.DataFrame({'all': g.score.mean(),
                      'complete yrs: FOMC in window': df[df.complete & df.fomc].groupby(['variant', 'mult']).score.mean(),
                      'complete yrs: no FOMC': df[df.complete & ~df.fomc].groupby(['variant', 'mult']).score.mean(),
                      'rates FOMC': df[df.complete & df.fomc & (df.cls == 'rates')].groupby(['variant', 'mult']).score.mean(),
                      'fx FOMC': df[df.complete & df.fomc & (df.cls == 'fx')].groupby(['variant', 'mult']).score.mean(),
                      '2021-24': df[df.origin >= '2021'].groupby(['variant', 'mult']).score.mean()})
    print(t.round(4).to_string())
    m0 = df.drop_duplicates(['name', 'origin'])
    print('M0: all %.4f | FOMC (complete) %.4f | no FOMC (complete) %.4f' % (
        m0.m0.mean(), m0[m0.complete & m0.fomc].m0.mean(), m0[m0.complete & ~m0.fomc].m0.mean()))
    print('cards with FOMC in window (complete years):', int((m0.complete & m0.fomc).sum()), 'of', len(m0))
    print('%.0fs' % (time.time() - t0))
