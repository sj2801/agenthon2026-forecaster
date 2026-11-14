"""forecast_rationale.md: numbered derivation + adjustment ledger (required, never scored)."""

from __future__ import annotations

import numpy as np


def write_rationale(u, p, bb, X, llm, docs, excluded, status_engine, n, secs, stress=None) -> str:
    L = [f"# Forecast rationale — {u.unit_id}", "",
         f"As-of {u.asof}. Target type `{u.target_type}` ({u.target_frequency}). "
         f"Assets {', '.join(u.assets)}; horizons {u.horizons}. {n} joint draws.", ""]
    L += ["## 1. Statistical backbone (text-blind)", ""]
    if bb is None:
        L += [f"Engine status: {status_engine}. A conservative driftless Gaussian walk "
              "(1.15x full-history step sd) was written instead.", ""]
    else:
        L += ["Per-step increments (level differences, or log(1+r) for return targets), "
              f"gap-spanning steps dropped. Variance term structure v(k)=s2_lr+(s2_now-s2_lr)*phi^k "
              f"with EWMA half-life {p.ewma_hl if u.target_frequency!='monthly' else p.m_ewma_hl} steps, "
              f"variance mean-reversion half-life {p.vol_hl if u.target_frequency!='monthly' else p.m_vol_hl} steps, "
              f"long-run window {p.lr_window if u.target_frequency!='monthly' else p.m_lr_window} steps. "
              f"Joint path covariance across assets and horizons; Student-t scale mixture nu={p.nu}.", ""]
        for a in u.assets:
            i = bb.info["assets"][a]
            L.append(f"- **{a}**: anchor {i['anchor']:.6g} (last obs {i['last_obs']}), step sd now/long-run "
                     f"{i['sd_now']:.4g}/{i['sd_lr']:.4g} ({i['sd_now']/max(i['sd_lr'],1e-12):.2f}x), "
                     f"drift/step {i['mu_step']:.3g}{' — STALE history, width inflated' if i['stale'] else ''}")
        L.append(f"- overall width x{p.vol_mult:.2f} (scheduled-event allowance); the card family "
                 f"({u.family or 'unknown'}) does not change the width "
                 f"(x{bb.info.get('family_mult', 1.0):.2f})")
        ew = bb.info.get("event_weights")
        if isinstance(ew, dict) and any(abs(w - 1) > 0.05 for v in ew.values() for w in v):
            L.append("- scheduled FOMC decision inside the window: per-step variance weights, measured "
                     f"from each series' own history around past decisions: {ew}")
        if len(u.assets) > 1:
            L.append(f"- step correlation (blend of long-window and EWMA): {bb.info['corr']}")
        L.append("")
    L += ["## 2. Text corpus", "",
          f"{len(docs)} documents dated <= as-of were read ({excluded} excluded by the local cutoff check)."]
    for d in docs[-15:]:
        L.append(f"- [{d.doc_id}] {d.timestamp} {d.doc_type}")
    L.append("")
    if llm is None:
        L += ["Text layer not used (disabled or failed); the forecast is the text-blind backbone.", ""]
    else:
        L += [f"House model status: {llm.status}; requests used: {llm.n_requests}.", ""]
        if llm.per_asset:
            L += ["Per-asset views (median over samples; direction in horizon-sd units, then shrunk x0.5; "
                  "vol multiplier shrunk toward 1 in log space; an *inferred* view never narrows):", ""]
            for a, r in llm.per_asset.items():
                L.append(f"- **{a}** [{r['basis']}]: direction samples {r['direction_samples']} -> applied "
                         f"{r['direction_applied']:+.3f} sd; vol samples {r['vol_samples']} -> applied "
                         f"x{r['vol_applied']:.3f}. Why: {' | '.join(w for w in r['why'] if w)[:600]}")
            L.append("")
        if llm.scenarios:
            L += ["Scenario mixture (recentred on the shrunk directional view; never narrower than the engine):", ""]
            for c in llm.scenarios:
                L.append(f"- {c[3] or 'scenario'}: p={c[0]:.2f}, vol x{c[2]:.2f}, shifts "
                         + ", ".join(f"{k} {v:+.2f}sd" for k, v in c[1].items()))
            L.append("")
        if llm.notes:
            L += ["Model summaries:", ""] + [f"> {s}" for s in llm.notes if s] + [""]
    if stress:
        ps, sz, dirs = stress
        sides = ", ".join(f"{a} {'down' if v < 0 else 'up' if v > 0 else 'none'}" for a, v in dirs.items())
        L += ["### Shock tail", "",
              f"{ps:.0%} of draws carry an extra {sz:.1f}-sd move in the stress direction ({sides}). "
              "Direction: fixed risk-off prior for FX/equity/labour series (havens appreciate, carry and "
              "equities fall). Rates and CPI carry no directional tail. Applied on tail/shock cards only.", ""]
    if llm is not None and getattr(llm, "event", False):
        L += [f"Scheduled event inside the window (majority of House samples): {llm.event_text or 'yes'}", ""]
    L += ["## 3. Resulting distribution", ""]
    cells = [(a, h) for a in u.assets for h in u.horizons]
    for j, (a, h) in enumerate(cells):
        q = np.quantile(X[:, j], [0.01, 0.05, 0.5, 0.95, 0.99])
        L.append(f"- {a} h={h}: median {q[2]:.5g}; 90% [{q[1]:.5g}, {q[3]:.5g}]; 98% [{q[0]:.5g}, {q[4]:.5g}]")
    L += ["", "## 4. Adjustment ledger", "",
          "| step | what | size | support |", "|---|---|---|---|",
          "| 1 | anchor | last observation (0 for return targets) | panel |",
          "| 2 | width | regime-vol term structure, t tails | panel, tuned on 2000-2024 rolling backtests |"]
    if llm is not None and llm.per_asset:
        for a, r in llm.per_asset.items():
            L.append(f"| 3 | {a} centre | {r['direction_applied']:+.3f} sd | {r['basis']} (House) |")
            L.append(f"| 4 | {a} width | x{r['vol_applied']:.3f} | {r['basis']} (House) |")
    if stress:
        L.append(f"| 5 | shock tail | {stress[0]:.0%} of draws, {stress[1]:.1f} sd | card family + risk-off prior / House tail side |")
    L += ["", "## 5. What would change this forecast", "",
          "A new policy decision or data surprise inside the window, or a shift in realised volatility; "
          "the width tracks the volatility regime and the centre moves only by bounded, shrunk text views.",
          "", f"_Runtime {secs:.1f}s._", ""]
    return "\n".join(L)
