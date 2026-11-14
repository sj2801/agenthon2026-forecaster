"""Loading a Track-2 unit: card, spec, panels, per-asset histories and step series.

Everything here is pure input handling; no forecasting decisions are made in this module.
"""

from __future__ import annotations

import json
import pathlib
import tomllib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

ASSET_COLS = ("asset", "asset_id")


@dataclass
class Unit:
    unit_id: str
    asof: str
    card: dict[str, Any]
    spec: dict[str, Any] | None
    card_path: pathlib.Path
    assets: list[str]
    horizons: list[int]
    target_type: str
    target_frequency: str
    family: str
    panels: dict[str, pd.DataFrame]
    text_dir: pathlib.Path | None = None
    # (n_assets, n_horizons) integer panel-step counts
    steps: np.ndarray | None = None
    notes: dict[str, str] = field(default_factory=dict)


def find_card(panels_dir: pathlib.Path, explicit: pathlib.Path | None = None) -> pathlib.Path:
    if explicit is not None:
        return explicit
    for cand in (panels_dir / "card.toml", panels_dir.parent / "card.toml"):
        if cand.exists():
            return cand
    raise SystemExit(f"card.toml not found in {panels_dir} or {panels_dir.parent}")


def read_panels(panels_dir: pathlib.Path) -> dict[str, pd.DataFrame]:
    found = sorted(panels_dir.glob("*.parquet"))
    if not found and panels_dir.parent.is_dir():
        found = sorted(panels_dir.parent.glob("*.parquet"))
    if not found:
        raise SystemExit(f"no .parquet under {panels_dir} (or its parent)")
    out = {}
    for p in found:
        df = pd.read_parquet(p)
        col = next((c for c in ASSET_COLS if c in df.columns), None)
        if col is None or "value" not in df.columns or "date" not in df.columns:
            continue
        df = df.rename(columns={col: "asset"})[["date", "asset", "value"]].copy()
        df["date"] = df["date"].astype(str).str.slice(0, 10)
        df["asset"] = df["asset"].astype(str)
        out[p.stem] = df
    return out


def series(panels: dict[str, pd.DataFrame], asset: str, asof: str) -> pd.Series:
    """History of one asset at or before as-of, from the first panel (sorted) that carries it."""
    for name in sorted(panels):
        df = panels[name]
        sub = df[(df["asset"] == asset) & (df["date"] <= asof)]
        if sub.empty:
            continue
        sub = sub.sort_values("date").drop_duplicates("date", keep="last")
        s = sub.set_index("date")["value"].astype(float)
        s = s[np.isfinite(s.to_numpy())]
        if len(s):
            return s
    raise SystemExit(f"asset {asset!r} not in any panel at or before {asof}")


def steps_of(s: pd.Series, target_type: str) -> pd.Series:
    """Per-step increments indexed by the date of the row they end on; gap-spanning steps -> NaN."""
    if target_type == "log_return":
        v = s.to_numpy()
        v = np.where(v <= -0.999, -0.999, v)
        d = pd.Series(np.log1p(v), index=s.index)
    else:
        d = s.diff()
    when = pd.to_datetime(pd.Series(s.index, index=s.index), errors="coerce")
    gap = when.diff().dt.days
    if gap.notna().sum():
        thr = max(float(gap.median()) * 10.0, 5.0)
        d = d.where(gap <= thr)
    return d


def load_unit(
    panels_dir: pathlib.Path,
    asof: str,
    text_dir: pathlib.Path | None,
    card_path: pathlib.Path | None = None,
) -> Unit:
    cp = find_card(panels_dir, card_path)
    card = tomllib.loads(cp.read_text())
    spec = None
    sp = cp.parent / "forecast_spec.json"
    if sp.exists():
        try:
            spec = json.loads(sp.read_text())
        except Exception:
            spec = None
    tgt = card["targets"]
    meta = card.get("metadata", {})
    u = Unit(
        unit_id=card["task"]["id"],
        asof=asof,
        card=card,
        spec=spec,
        card_path=cp,
        assets=[str(a) for a in tgt["asset_ids"]],
        horizons=[int(h) for h in tgt["horizons"]],
        target_type=str(tgt.get("target_type", meta.get("target_type", "level"))),
        target_frequency=str(tgt.get("target_frequency", meta.get("target_frequency", "daily"))),
        family=str(meta.get("category", "")),
        panels=read_panels(panels_dir),
        text_dir=text_dir,
    )
    u.notes = {
        "title": str(card["task"].get("title", "")),
        "description": str(meta.get("description", "")),
        "text_notes": str(card.get("text", {}).get("notes", "")),
    }
    u.steps = step_counts(u)
    return u


def step_counts(u: Unit) -> np.ndarray:
    """Panel steps per (asset, horizon): the declared horizon on daily cards, calendar months on
    monthly cards (same rule as M0 / the reference CLI)."""
    base = np.tile(np.array(u.horizons, dtype=int), (len(u.assets), 1))
    if u.target_frequency != "monthly":
        return base
    try:
        from qfbench2_track_forecasting.cli import _monthly_inputs

        st = _monthly_inputs(u.panels_raw if hasattr(u, "panels_raw") else _raw_panels(u),
                             u.card, u.card_path, u.asof)
        if st is not None:
            return np.asarray(st, dtype=int).reshape(len(u.assets), len(u.horizons))
    except BaseException:
        pass
    # Fallback: months from last observation to (asof + horizon business days).
    out = base.copy()
    for ai, a in enumerate(u.assets):
        s = series(u.panels, a, u.asof)
        last = pd.Period(s.index[-1], freq="M")
        for hi, h in enumerate(u.horizons):
            tgt = pd.Period(pd.Timestamp(u.asof) + pd.offsets.BDay(h), freq="M")
            out[ai, hi] = max(1, (tgt - last).n)
    return out


def _raw_panels(u: Unit) -> dict[str, pd.DataFrame]:
    # the reference helper expects the original column names; ours are compatible
    return {k: v.copy() for k, v in u.panels.items()}
