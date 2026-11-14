"""Exact re-implementation of the organizer baseline M0 (docs/M0-BASELINE.md §3, §5).

Used for two things:
  * the expected-error divisors that normalize every score (so local backtests are on the
    official scale), and
  * a reference forecast to compare against.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
from scipy.special import gamma as G
from scipy.special import hyp1f1
from scipy.stats import norm

TAIL_LEVELS = (0.01, 0.05, 0.95, 0.99)


def _m0_steps(s: pd.Series, target_type: str, window: int = 300) -> pd.Series:
    s = s.iloc[-window:]
    if target_type == "log_return":
        d = pd.Series(s.to_numpy(dtype=float), index=s.index)  # the row itself (§3.2)
    else:
        d = s.diff()
    when = pd.to_datetime(pd.Series(s.index, index=s.index))
    gap = when.diff().dt.days
    med = float(gap.median()) if gap.notna().any() else 1.0
    thr = max(10.0 * med, 5.0)
    d = d.where(gap <= thr)  # row 0 has NaN gap -> dropped
    return d.dropna()


def m0_mean_cov(hist: dict[str, pd.Series], assets: list[str], steps: np.ndarray,
                target_type: str, window: int = 300):
    """mean (d,), cov C (d,d) in card grid order (asset-major, horizon-minor)."""
    st = {a: _m0_steps(hist[a], target_type, window) for a in assets}
    frame = pd.concat(st, axis=1, join="inner").dropna()
    order = sorted(assets)
    X = frame[order].to_numpy(dtype=float)
    mu = dict(zip(order, X.mean(axis=0)))
    if X.shape[1] == 1:
        Sig = np.array([[np.var(X[:, 0], ddof=1)]])
    else:
        Sig = np.cov(X, rowvar=False)
    idx = {a: i for i, a in enumerate(order)}
    cells = [(a, int(steps[ai, hi])) for ai, a in enumerate(assets) for hi in range(steps.shape[1])]
    d = len(cells)
    mean = np.empty(d)
    cov = np.empty((d, d))
    for i, (a, s) in enumerate(cells):
        anchor = 0.0 if target_type == "log_return" else float(hist[a].iloc[-1])
        mean[i] = anchor + s * mu[a]
        for j, (b, t) in enumerate(cells):
            cov[i, j] = min(s, t) * Sig[idx[a], idx[b]]
    cov = cov + 1e-10 * np.eye(d)
    C = cov + 1e-9 * np.eye(d)
    try:
        L = np.linalg.cholesky(C)
    except np.linalg.LinAlgError:
        C = np.diag(np.diag(C))
        L = np.sqrt(C)
    return mean, C, L


def m0_draws(unit_id: str, mean: np.ndarray, L: np.ndarray, n: int = 500) -> np.ndarray:
    seed = zlib.crc32(unit_id.encode()) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, len(mean)))
    return mean + Z @ L.T


def _e_abs(delta, s):
    return s * np.sqrt(2 / np.pi) * np.exp(-delta**2 / (2 * s**2)) + delta * (1 - 2 * norm.cdf(-delta / s))


def _e_sqrt_abs(delta, s):
    return np.sqrt(s) * 2**0.25 * G(0.75) / np.sqrt(np.pi) * hyp1f1(-0.25, 0.5, -delta**2 / (2 * s**2))


def m0_divisors(mean: np.ndarray, C: np.ndarray, tail_levels=TAIL_LEVELS) -> dict[str, float]:
    sd = np.sqrt(np.diag(C))
    marg = float(np.mean(sd) / np.sqrt(np.pi))
    phi_mean = float(np.mean(norm.pdf(norm.ppf(np.asarray(tail_levels)))))
    tail = phi_mean * float(np.mean(sd))
    d = len(mean)
    joint = 0.0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            delta = mean[i] - mean[j]
            s2 = C[i, i] + C[j, j] - 2 * C[i, j]
            if s2 <= 0:
                continue
            s = np.sqrt(s2)
            joint += float(_e_abs(delta, s) - _e_sqrt_abs(delta, s) ** 2)
    out = {"marginal": marg, "joint": joint, "tail": tail}
    return {k: (v if v > 0 else 1.0) for k, v in out.items()}


# --------------------------------------------------------------------------- scoring
def _crps(samples, y):
    from qfbench2_common.scoring import crps

    return crps.crps_marginal(samples, y)


def normalized_composite(samples: np.ndarray, y: np.ndarray, div: dict[str, float],
                         tail_levels=TAIL_LEVELS, clip: float = 8.0) -> dict[str, float]:
    """Official per-card normalized composite (with single-cell weight redistribution)."""
    from qfbench2_common.scoring import crps
    from qfbench2_track_forecasting.tail import tail_pinball

    m = crps.crps_marginal(samples, y) / div["marginal"]
    t = tail_pinball(samples, y, tail_levels) / div["tail"]
    if samples.shape[1] == 1:
        comp = 0.714 * m + 0.286 * t
        j = 0.0
    else:
        j = crps.variogram_score(samples, y, p=0.5) / div["joint"]
        comp = 0.5 * m + 0.3 * j + 0.2 * t
    return {"marginal": m, "joint": j, "tail": t, "composite": min(float(comp), clip)}
