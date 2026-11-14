"""`forecast` verb: text-conditioned, fat-tailed joint forecaster for Agenthon 2026 Track 2.

    forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD --out /output/forecast.parquet

Writes forecast.parquet, forecast_meta.json and forecast_rationale.md next to --out.
Never fails silently: if anything in the text layer breaks, the text-blind engine is used; if the
engine itself breaks, a conservative random walk is written so the card is never left unscored.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
import traceback

import numpy as np
import pandas as pd

from .corpus import digest, read_corpus
from .data import load_unit, series
from .engine import Adjust, Params, build_backbone, default_seed, sample, stress_adjust
from .params import TUNED, card_overrides

N_DRAWS = 4000


def _recent_changes(u, hist) -> dict:
    out = {}
    for a in u.assets:
        s = hist[a]
        parts = []
        for k in ((1, 3, 6) if u.target_frequency == "monthly" else (5, 21, 63)):
            if len(s) > k:
                if u.target_type == "log_return":
                    v = float(np.sum(np.log1p(s.iloc[-k:].to_numpy())))
                    parts.append(f"{k}-step cum log-ret {v:+.4f}")
                else:
                    parts.append(f"{k}-step {float(s.iloc[-1] - s.iloc[-1 - k]):+.4g}")
        out[a] = {"chg": "; ".join(parts)}
    return out


def _fallback(u, n, seed):
    """Last-resort: driftless Gaussian walk per cell from full-history step sd."""
    rng = np.random.default_rng(seed)
    cols = []
    for ai, a in enumerate(u.assets):
        s = series(u.panels, a, u.asof)
        st = (np.log1p(s.to_numpy()) if u.target_type == "log_return" else np.diff(s.to_numpy()))
        sd = float(np.nanstd(st[-750:])) or 1e-6
        z = rng.standard_normal(n)
        for hi in range(len(u.horizons)):
            h = int(u.steps[ai, hi])
            anchor = 0.0 if u.target_type == "log_return" else float(s.iloc[-1])
            cols.append(anchor + 1.15 * sd * np.sqrt(h) * z)
    return np.column_stack(cols)


def _main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "forecast":  # tolerate the verb as a leading positional
        argv = argv[1:]
    ap = argparse.ArgumentParser(prog="forecast")
    ap.add_argument("--panels", type=pathlib.Path, required=True)
    ap.add_argument("--text", type=pathlib.Path, required=True)
    ap.add_argument("--asof", required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--card", type=pathlib.Path, default=None)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--no-llm", action="store_true")
    a, unknown = ap.parse_known_args(argv)  # tolerate extra flags a launcher may add
    t0 = time.time()
    log = lambda *x: print(*x, file=sys.stderr, flush=True)  # noqa: E731

    u = load_unit(a.panels, a.asof, a.text, a.card)
    floor = int(u.card.get("scoring", {}).get("params", {}).get("n_draws_min", 200) or 200)
    n = int(min(max(a.n_draws, floor, 500), 20000))
    seed = default_seed(u.unit_id, os.environ.get("QFBENCH_SEED"))
    p = Params(**{**TUNED, **card_overrides(u.assets)})
    hist = {x: series(u.panels, x, u.asof) for x in u.assets}

    status_engine, llm_res, docs, excluded, stress_used = "ok", None, [], 0, None
    try:
        bb = build_backbone(u, p, hist)
        if not np.all(np.isfinite(bb.mean)) or not np.all(np.isfinite(bb.cov)):
            raise ValueError("non-finite backbone")
        adj = Adjust()
        try:
            docs, excluded = read_corpus(a.text, a.asof)
            if not a.no_llm:
                from .llm import query
                dig = digest(docs, u.assets, " ".join(u.notes.values()))
                llm_res = query(u, bb, dig, _recent_changes(u, hist), log=log)
                adj = llm_res.adjust
        except Exception as e:  # text layer must never sink the card
            log("text layer failed:", repr(e))
            traceback.print_exc()
            adj = Adjust()
            llm_res = None
        try:
            from .llm import USE_TEXT_TAIL
            use_tx = USE_TEXT_TAIL and llm_res is not None
            sides = llm_res.tail_sides if use_tx else {}
            ev = bool(llm_res.event) if llm_res is not None else False
            unan = llm_res.tail_unanimous if use_tx else {}
            adj.stress = stress_adjust(u, p, sides, ev, unan)
        except Exception as e:
            log("stress layer skipped:", repr(e))
            adj.stress = None
        stress_used = adj.stress
        X = sample(bb, p, n, seed, adj)
        if not np.all(np.isfinite(X)):
            raise ValueError("non-finite draws")
    except Exception as e:
        log("engine failed, writing fallback:", repr(e))
        traceback.print_exc()
        status_engine = f"fallback ({type(e).__name__})"
        bb = None
        X = _fallback(u, n, seed)

    out_dir = a.out.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cells = [(asset, h) for asset in u.assets for h in u.horizons]
    df = pd.DataFrame({
        "draw": np.repeat(np.arange(n, dtype=np.int32), len(cells)),
        "asset": np.tile(np.array([c[0] for c in cells], dtype=object), n),
        "horizon": np.tile(np.array([c[1] for c in cells], dtype=np.int32), n),
        "value": X.reshape(-1).astype(np.float64),
    })
    df["asset"] = df["asset"].astype("string")
    df.to_parquet(a.out, index=False)

    meta = {"unit_id": u.unit_id, "asof": a.asof, "representation": "samples",
            "asset_ids": u.assets, "horizons": u.horizons, "n_draws": n,
            "target": u.target_type,
            "rationale": {"file": "forecast_rationale.md",
                          "method": ("regime-vol joint Student-t path model"
                                     + ("" if os.environ.get("T2_NO_LLM") else " + House-LLM bounded text adjustments"))}}
    (out_dir / "forecast_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    from .rationale import write_rationale
    (out_dir / "forecast_rationale.md").write_text(
        write_rationale(u, p, bb, X, llm_res, docs, excluded, status_engine, n, time.time() - t0,
                        stress_used))
    log(f"wrote {n} draws x {len(cells)} cells for {u.unit_id} in {time.time()-t0:.1f}s; "
        f"text: {llm_res.status if llm_res else 'not used'}")
    return 0


def _reference_fallback(argv) -> int:
    """Last resort: the organizer's reference forecaster (qfbench2_track_forecasting, M0-style joint
    Gaussian walk), run with the same arguments. It understands every card format the reference
    package supports, so an unfamiliar card still gets a valid, scoreable forecast instead of 8.0."""
    from qfbench2_track_forecasting.cli import main as reference_main

    keep = []
    it = iter(argv)
    for tok in it:
        if tok in ("--panels", "--text", "--asof", "--out", "--card"):
            keep += [tok, next(it)]
        elif tok.split("=", 1)[0] in ("--panels", "--text", "--asof", "--out", "--card") and "=" in tok:
            keep.append(tok)
    return int(reference_main(keep) or 0)


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "forecast":
        argv = argv[1:]
    try:
        return _main(argv)
    except SystemExit as e:  # argparse --help / usage errors keep their normal behaviour
        if e.code in (0, None, 2) and not any(t == "--out" for t in argv):
            raise
        err = e
    except BaseException as e:  # noqa: BLE001 - nothing may leave the card unscored
        err = e
    print(f"agent failed ({err!r}); writing the organizer reference forecast instead", file=sys.stderr, flush=True)
    traceback.print_exc()
    return _reference_fallback(argv)


if __name__ == "__main__":
    sys.exit(main())
