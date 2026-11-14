"""Score claude_final variants and M0 on the 22 short-horizon practice units (M0-normalized)."""
import sys, glob, pathlib
sys.path.insert(0, '.')
import t2agent.engine as E
sys.path.append('research')
import numpy as np, pandas as pd
from t2agent.data import load_unit, series
from t2agent.engine import Adjust, Params, build_backbone, sample, stress_adjust, default_seed
from t2agent.params import TUNED
from m0lib import m0_mean_cov, m0_divisors, normalized_composite
R = pathlib.Path('../track2-forecasting-public')
OLD = {'fam_mult': {"T2-F1": 1.0, "T2-F2": 1.75, "T2-F3": 1.15, "T2-F4": 2.5}, 'stress_tail': {"T2-F4": (0.30, 1.5)}, 'nu': 5.0, 'vol_mult': 1.0, 'event_shrink': 0.0}
V = {'claude_final': {}, 'no events': {'event_shrink': 0.0}, 'events 1.0': {'event_shrink': 1.0},
     'vm1.00': {'vol_mult': 1.0}, 'vm1.15': {'vol_mult': 1.15}, 'nu4': {'nu': 4.0}, 'ewma10': {'ewma_hl': 10.0},
     'claude2-nollm': OLD}
rows = []
for ud in sorted(R.glob('units/*-ps1')):
    uid = ud.name
    asof = [l.split('"')[1] for l in open(ud / 'card.toml') if l.strip().startswith('data_cutoff')][0]
    u = load_unit(ud / 'panels', asof, ud / 'text')
    hist = {a: series(u.panels, a, u.asof) for a in u.assets}
    ans = pd.read_parquet(R / 'practice-answers' / f'{uid}.parquet')
    y = np.array([float(ans[(ans.asset == a) & (ans.horizon == h)].value.iloc[0]) for a in u.assets for h in u.horizons])
    mean, C, L = m0_mean_cov(hist, u.assets, u.steps, u.target_type)
    div = m0_divisors(mean, C)
    m0f = pd.read_parquet(f'.cache/ps1/m0/{uid}/forecast.parquet')
    M = m0f.pivot_table(index='draw', columns=['asset', 'horizon'], values='value')[[(a, h) for a in u.assets for h in u.horizons]].to_numpy()
    r = dict(unit=uid, fam=u.family, cls=('rates' if u.assets[0].startswith('UST') else ('em' if u.assets[0] in ('MXN', 'BRL', 'ZAR', 'KRW', 'INR') else 'g10')),
             M0=normalized_composite(M, y, div)['composite'])
    for name, over in V.items():
        p = Params(**{**TUNED, **over})
        bb = build_backbone(u, p, hist)
        X = sample(bb, p, 4000, default_seed(u.unit_id), Adjust(stress=stress_adjust(u, p)))
        r[name] = normalized_composite(X, y, div)['composite']
        if name == 'claude_final':
            ew = bb.info.get('event_weights'); r['event'] = isinstance(ew, dict) and any(abs(w - 1) > .05 for v in ew.values() for w in v)
            # PIT of realized values in the forecast
            r['pit'] = np.round([(X[:, j] < y[j]).mean() for j in range(len(y))], 2).tolist()
    rows.append(r)
df = pd.DataFrame(rows); pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 40)
cols = ['M0'] + list(V)
print(df[['unit', 'fam', 'cls', 'event'] + cols + ['pit']].round(3).to_string(index=False))
print('\nMEAN'); print(df[cols].mean().round(4).to_string())
print('\nby class'); print(df.groupby('cls')[cols].mean().round(3).to_string())
print('\nwins vs M0 (claude_final):', int((df.claude_final < df.M0).sum()), 'of', len(df))
df.to_pickle('.cache/ps1eval.pkl')
