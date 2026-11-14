"""Text-blind probabilistic engine (the numerical backbone of the agent).

Model, per card
---------------
Per-step increments x_t of each target series (level differences, or log(1+r) for
`log_return` targets) are modelled as

    x_t = mu + sqrt(v(k)) * eps_t ,  eps_t ~ multivariate (correlation R), fat-tailed

with a *mean-reverting variance term structure*: the per-step variance k steps ahead is

    v(k) = s2_lr + (s2_now - s2_lr) * phi**k

s2_now is an EWMA variance (regime), s2_lr a long-window variance, phi the per-step persistence
of variance shocks (GARCH-style).  The h-step change is the sum of increments, so for cells
i=(a,h_i), j=(b,h_j) the covariance is the exact path covariance

    Cov = R_ab * sum_{k<=min(h_i,h_j)} sqrt(v_a(k) v_b(k))

which keeps the cross-horizon structure the variogram rewards.  Draws are a scale mixture
(multivariate Student-t with nu dof, variance preserved), and an optional scenario mixture from
the text layer shifts centres / rescales spread per scenario.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .data import Unit, series, steps_of


@dataclass
class Params:
    lr_window: int = 750          # steps used for long-run variance / correlation / drift
    ewma_hl: float = 30.0         # half-life (steps) of the regime variance
    vol_hl: float = 60.0          # half-life (steps) of variance mean reversion
    corr_hl: float = 120.0        # half-life of EWMA correlation
    corr_shrink: float = 0.5      # weight on long-window correlation
    nu: float = 6.0               # Student-t dof of the joint scale mixture (inf -> Gaussian)
    drift_level: float = 0.0      # shrink factor on long-window mean step, level targets
    drift_logret: float = 0.5     # shrink factor on long-window mean step, log-return targets
    # monthly panels (macro): fewer obs, strongly trending series
    m_lr_window: int = 120
    m_ewma_hl: float = 6.0
    m_vol_hl: float = 6.0
    m_drift_window: int = 12
    m_drift_shrink: float = 1.0
    stale_vol_mult: float = 1.25  # extra width when the target history is stale (transfer cards)
    vol_mult: float = 1.0         # global calibration multiplier on sd
    # Card-family width prior. Cards are authored around events; F4 = tail/shock-from-text,
    # F2 = text-cued regime shift. Cross-validated across the public-dev / validation splits.
    fam_mult: dict = field(default_factory=lambda: {"T2-F1": 1.0, "T2-F2": 1.75,
                                                    "T2-F3": 1.15, "T2-F4": 2.5})
    # Shock tail for tail/shock cards: with probability p a draw is moved s engine-sd in the
    # asset's stress direction (RISK_OFF below, or the side the text layer names). Chosen
    # identically by both practice splits; p=0.30 trades gain against wrong-direction cost.
    stress_tail: dict = field(default_factory=lambda: {"T2-F4": (0.30, 1.5)})
    # Milder tail for F4 assets with no fixed risk-off sign (rates, CPI), used only when every
    # House sample names the same side: a wrong side costs about 1.6x what a right side gains.
    text_tail: tuple = (0.15, 1.0)
    # Directional tails on other card families: off. Measured break-even side accuracy is ~64%;
    # scheduled events there widen the distribution instead (text layer), never skew it.
    event_tail: tuple | None = None
    # Scheduled-event variance (FOMC decision days). 0 = off. Otherwise each series' own history
    # gives the variance ratio of the step that contains a decision (same-day close) and of the
    # first step after it (next-day fixings); the forecast step that will contain the next
    # decision gets that extra variance, shrunk toward 1 by this factor.
    event_shrink: float = 0.0


    def to_dict(self):
        return asdict(self)


# Sign of the value move in a risk-off shock, from the quote conventions of the panels
# (H.10: EUR/GBP/AUD/NZD are USD per currency, the rest currency per USD; JKP factor returns).
# A fixed economic prior: equities and carry fall, havens appreciate, labour data weaken.
# Rates and CPI are deliberately absent: their shock side (hawkish vs flight-to-quality) is
# left to the text layer.
RISK_OFF = {
    "MKT": -1, "SMB": -1, "MOM": -1, "HML": -1, "BAB": -1, "QMJ": +1,
    "EUR": -1, "GBP": -1, "AUD": -1, "NZD": -1,
    "JPY": -1, "CHF": -1,
    "CAD": +1, "NOK": +1, "SEK": +1, "DKK": +1, "CNY": +1, "INR": +1, "BRL": +1,
    "NFP": -1, "UNRATE": +1,
}


@dataclass
class Adjust:
    """Bounded text-driven adjustments (identity = no text)."""
    vol_mult: dict[str, float] = field(default_factory=dict)        # per asset
    # scenarios: list of (prob, {asset: shift_z}, vol_mult)
    scenarios: list[tuple[float, dict[str, float], float]] = field(default_factory=list)
    # stress tail: (prob, shift_z, {asset: direction in {-1,0,+1}}); independent of scenarios
    stress: tuple | None = None


def _ewma_var(x: np.ndarray, hl: float) -> float:
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(len(x))[::-1]
    w /= w.sum()
    return float(np.sum(w * x**2))


def _ewma_cov(X: np.ndarray, hl: float) -> np.ndarray:
    lam = 0.5 ** (1.0 / hl)
    w = lam ** np.arange(len(X))[::-1]
    w /= w.sum()
    Xc = X - (w[:, None] * X).sum(0)
    return (w[:, None] * Xc).T @ Xc


def _corr(C: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.clip(np.diag(C), 1e-300, None))
    R = C / np.outer(s, s)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -0.999, 0.999)


def _nearest_psd_corr(R: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh((R + R.T) / 2)
    w = np.clip(w, 1e-6, None)
    R2 = V @ np.diag(w) @ V.T
    s = np.sqrt(np.diag(R2))
    R2 = R2 / np.outer(s, s)
    np.fill_diagonal(R2, 1.0)
    return R2


def _event_ratios(x: pd.Series, dates: np.ndarray, hl: float = 20.0, min_events: int = 20):
    """Variance ratios of the step ending on a decision day (lag 0) and of the first step after it
    (lag 1), against ordinary steps, from EWMA-standardised increments; plus their frequencies."""
    x = x.dropna()
    if len(x) < 500:
        return 1.0, 1.0, 0.0, 0.0
    idx = pd.to_datetime(pd.Series(x.index)).to_numpy(dtype="datetime64[D]")
    v = x.to_numpy(dtype=float)
    v = v - v.mean()
    lam = 0.5 ** (1.0 / hl)
    s2 = np.empty(len(v))
    s2[0] = np.var(v[:60]) + 1e-300
    for t in range(1, len(v)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * v[t - 1] ** 2
    z2 = v**2 / np.maximum(s2, 1e-300)
    z2, idx = z2[60:], idx[60:]
    ev = dates[(dates >= idx[0]) & (dates <= idx[-1])]
    lag0 = np.isin(idx, ev)
    prev = np.concatenate([[idx[0]], idx[:-1]])
    # first observation strictly after a decision day (decision day itself may be absent)
    j = np.searchsorted(idx, ev, side="right")
    lag1 = np.zeros(len(idx), bool)
    lag1[j[j < len(idx)]] = True
    lag1 &= ~lag0
    normal = ~(lag0 | lag1)
    if normal.sum() < 200:
        return 1.0, 1.0, 0.0, 0.0
    base = float(np.mean(z2[normal]))
    r0 = float(np.mean(z2[lag0]) / base) if lag0.sum() >= min_events else 1.0
    r1 = float(np.mean(z2[lag1]) / base) if lag1.sum() >= min_events else 1.0
    return r0, r1, float(lag0.mean()), float(lag1.mean())


def _event_weights(u: Unit, inc: dict, kmax: int, shrink: float) -> dict:
    """Per-asset weights (length kmax) for the per-step variance path."""
    from .events import FOMC_DECISIONS

    dates = np.array(FOMC_DECISIONS, dtype="datetime64[D]")
    asof = np.datetime64(pd.Timestamp(u.asof).date(), "D")
    steps = np.array([np.busday_offset(asof, k, roll="forward") for k in range(0, kmax + 1)], dtype="datetime64[D]")
    out = {}
    for a in u.assets:
        r0, r1, f0, f1 = _event_ratios(inc[a], dates)
        r0 = float(np.clip(1 + shrink * (r0 - 1), 0.5, 4.0))
        r1 = float(np.clip(1 + shrink * (r1 - 1), 0.5, 4.0))
        c = 1 + (r0 - 1) * f0 + (r1 - 1) * f1
        w = np.ones(kmax)
        for k in range(1, kmax + 1):
            lo, hi = steps[k - 1], steps[k]
            if np.any(dates == hi):
                w[k - 1] = r0
            elif np.any((dates >= lo) & (dates < hi)):
                w[k - 1] = r1
        out[a] = w / c
    return out


@dataclass
class Backbone:
    mean: np.ndarray        # (d,)
    cov: np.ndarray         # (d,d)
    sd: np.ndarray          # (d,)
    cells: list             # [(asset, horizon, steps)]
    info: dict


def build_backbone(u: Unit, p: Params, hist: dict[str, pd.Series] | None = None) -> Backbone:
    assets, steps = u.assets, u.steps
    monthly = u.target_frequency == "monthly"
    if hist is None:
        hist = {a: series(u.panels, a, u.asof) for a in assets}
    lr_w = p.m_lr_window if monthly else p.lr_window
    ewma_hl = p.m_ewma_hl if monthly else p.ewma_hl
    vol_hl = p.m_vol_hl if monthly else p.vol_hl

    inc = {}
    info = {"assets": {}}
    for a in assets:
        st = steps_of(hist[a], u.target_type).dropna()
        st = st.iloc[-max(lr_w, 60):]
        inc[a] = st
    s2_lr, s2_now, mu = {}, {}, {}
    for a in assets:
        x = inc[a].to_numpy(dtype=float)
        if len(x) < 5:
            x = np.concatenate([x, np.zeros(5 - len(x))]) if len(x) else np.zeros(5) + 1e-6
        lr = float(np.var(x, ddof=1)) if len(x) > 2 else float(np.mean(x**2))
        # regime variance from demeaned EWMA of squared increments
        m_lr = float(np.mean(x))
        now = _ewma_var(x - m_lr, ewma_hl)
        # staleness: last usable increment long before the as-of (transfer / EM cards)
        last_inc = pd.Timestamp(inc[a].index[-1]) if len(inc[a]) else pd.Timestamp(u.asof)
        stale_days = (pd.Timestamp(u.asof) - last_inc).days
        stale = stale_days > (75 if monthly else 30)
        if stale:
            now = lr * p.stale_vol_mult**2
            lr = lr * p.stale_vol_mult**2
        s2_lr[a], s2_now[a] = max(lr, 1e-16), max(now, 1e-16)
        if u.target_type == "log_return":
            mu[a] = p.drift_logret * m_lr
        elif monthly:
            mu[a] = p.m_drift_shrink * float(np.mean(x[-p.m_drift_window:]))
        else:
            mu[a] = p.drift_level * m_lr
        info["assets"][a] = {
            "anchor": 0.0 if u.target_type == "log_return" else float(hist[a].iloc[-1]),
            "last_obs": str(hist[a].index[-1]),
            "sd_lr": float(np.sqrt(s2_lr[a])),
            "sd_now": float(np.sqrt(s2_now[a])),
            "mu_step": float(mu[a]),
            "n_inc": int(len(inc[a])),
            "stale": bool(stale),
        }

    # correlation from date-aligned increments
    if len(assets) > 1:
        frame = pd.concat({a: inc[a] for a in assets}, axis=1, join="inner").dropna()
        if len(frame) >= 20:
            X = frame[assets].to_numpy(dtype=float)
            R_lr = _corr(np.cov(X, rowvar=False))
            R_now = _corr(_ewma_cov(X, p.corr_hl))
            R = p.corr_shrink * R_lr + (1 - p.corr_shrink) * R_now
            R = _nearest_psd_corr(R)
        else:
            R = np.eye(len(assets))
    else:
        R = np.eye(1)
    info["corr"] = R.round(3).tolist()

    phi = 0.5 ** (1.0 / vol_hl)
    kmax = int(steps.max())
    k = np.arange(1, kmax + 1)
    vpath = {a: s2_lr[a] + (s2_now[a] - s2_lr[a]) * phi**k for a in assets}  # per-step variance
    if p.event_shrink > 0 and not monthly:
        try:
            ev = _event_weights(u, inc, kmax, p.event_shrink)
            vpath = {a: vpath[a] * ev[a] for a in assets}
            info["event_weights"] = {a: [round(float(w), 3) for w in ev[a]] for a in assets}
        except Exception as e:  # never let the event layer break the forecast
            info["event_weights"] = f"skipped ({type(e).__name__})"

    cells = [(a, u.horizons[hi], int(steps[ai, hi])) for ai, a in enumerate(assets) for hi in range(len(u.horizons))]
    d = len(cells)
    mean = np.empty(d)
    cov = np.empty((d, d))
    aidx = {a: i for i, a in enumerate(assets)}
    for i, (a, _, s) in enumerate(cells):
        mean[i] = info["assets"][a]["anchor"] + s * mu[a]
        for j, (b, _, t) in enumerate(cells):
            m = min(s, t)
            cov[i, j] = R[aidx[a], aidx[b]] * float(np.sum(np.sqrt(vpath[a][:m] * vpath[b][:m])))
    fm = float(p.fam_mult.get(u.family, 1.0)) if p.fam_mult else 1.0
    info["family_mult"] = fm
    cov *= (p.vol_mult * fm) ** 2
    cov = (cov + cov.T) / 2 + 1e-12 * np.eye(d)
    return Backbone(mean=mean, cov=cov, sd=np.sqrt(np.diag(cov)), cells=cells, info=info)


def sample(bb: Backbone, p: Params, n: int, seed: int, adj: Adjust | None = None,
           assets: list[str] | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    d = len(bb.mean)
    try:
        L = np.linalg.cholesky(bb.cov)
    except np.linalg.LinAlgError:
        w, V = np.linalg.eigh(bb.cov)
        L = V @ np.diag(np.sqrt(np.clip(w, 0, None)))
    # antithetic normals for lower Monte-Carlo noise
    half = (n + 1) // 2
    Z = rng.standard_normal((half, d))
    Z = np.vstack([Z, -Z])[:n]
    X = Z @ L.T
    if np.isfinite(p.nu) and p.nu > 2:
        chi = rng.chisquare(p.nu, size=n)
        W = np.sqrt((p.nu - 2.0) / chi)
        X = X * W[:, None]
    cell_assets = [c[0] for c in bb.cells]
    # per-asset vol multipliers from text
    if adj is not None and adj.vol_mult:
        vm = np.array([adj.vol_mult.get(a, 1.0) for a in cell_assets])
        X = X * vm[None, :]
    out = bb.mean[None, :] + X
    if adj is not None and adj.scenarios:
        probs = np.array([max(s[0], 0.0) for s in adj.scenarios], dtype=float)
        probs = probs / probs.sum()
        # deterministic stratified assignment keeps the mixture weights exact
        counts = np.floor(probs * n).astype(int)
        counts[np.argmax(probs)] += n - counts.sum()
        lab = np.repeat(np.arange(len(probs)), counts)
        rng.shuffle(lab)
        for kk, (_, shifts, vmk) in enumerate(adj.scenarios):
            rows = lab == kk
            if not rows.any():
                continue
            shift = np.array([shifts.get(a, 0.0) for a in cell_assets]) * bb.sd
            out[rows] = bb.mean[None, :] + shift[None, :] + X[rows] * vmk
    if adj is not None and adj.stress:
        ps, sz, dirs = adj.stress
        dvec = np.array([float(dirs.get(a, 0)) for a in cell_assets])
        if ps > 0 and np.any(dvec != 0):
            k = int(round(ps * n))
            # separate stream so the base draws stay identical with and without the stress tail
            srng = np.random.default_rng(int(seed) ^ 0x5A5A5A5)
            rows = srng.permutation(n)[:k]
            out[rows] = out[rows] + (sz * dvec * bb.sd)[None, :]
    return out


def stress_adjust(u, p: Params, text_sides: dict | None = None, event: bool = False,
                  text_unanimous: dict | None = None):
    """The stress-tail layer for a card: family prior + static risk-off map + text-layer sides.

    F4 cards: every asset with a static risk-off sign uses it unless the text layer names the
    opposite side (then that asset goes symmetric rather than flipped); assets without a static
    sign (rates, CPI) take the text layer's side if it names one.
    Other cards: only when the text layer reports a scheduled event inside the window, and only
    on the sides it names, with the milder event_tail.
    """
    text_sides = text_sides or {}
    text_unanimous = text_unanimous or {}
    fam = p.stress_tail.get(u.family)
    dirs = {}
    if fam:
        static_used = False
        for a in u.assets:
            st = RISK_OFF.get(a, 0)
            tx = int(text_unanimous.get(a, 0))
            if st:
                dirs[a] = 0 if (tx and tx != st) else st      # unanimously opposed by the text -> symmetric
                static_used = static_used or bool(dirs[a])
            else:
                dirs[a] = int(text_unanimous.get(a, 0))       # rates / CPI: unanimous text side only
        if not any(dirs.values()):
            return None
        if static_used:
            return (fam[0], fam[1], dirs)
        return (p.text_tail[0], p.text_tail[1], dirs) if p.text_tail else None
    if event and p.event_tail and text_sides:
        dirs = {a: int(text_sides.get(a, 0)) for a in u.assets}
        if any(dirs.values()):
            return (p.event_tail[0], p.event_tail[1], dirs)
    return None


def default_seed(unit_id: str, env_seed: str | None = None) -> int:
    base = zlib.crc32(unit_id.encode()) & 0x7FFFFFFF
    if env_seed:
        try:
            return (base ^ int(env_seed)) & 0x7FFFFFFF
        except ValueError:
            pass
    return base
