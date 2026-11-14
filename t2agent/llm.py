"""Text layer: ask the House model for *bounded* distribution adjustments.

The model never produces draws.  It returns, per target asset, a directional view in units of
the engine's own horizon standard deviation, a volatility multiplier, and optionally 2-3
scenarios for a binary event inside the window.  Everything it returns is validated, clamped and
shrunk before use; any failure falls back to the text-blind engine and is recorded.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .engine import Adjust

# ---------------------------------------------------------------- policy (bounds & shrinkage)
MAX_DIR_Z = 0.6        # |direction| the model may state, in horizon-sd units
DIR_SHRINK = 0.5       # applied shift = DIR_SHRINK * median direction
VOL_LO, VOL_HI = 0.85, 1.6
VOL_SHRINK = 0.6       # applied log-vol = VOL_SHRINK * log(median vol_mult)
# v3: the v2 event boost (ceiling 1.8, shrink 0.85) is OFF. Board A/B: the v1 policy measured
# +0.029 vs its no-LLM twin, the v2 policy -0.012. Kept as named constants for clarity.
VOL_HI_EVENT = VOL_HI
VOL_SHRINK_EVENT = VOL_SHRINK
# v3: text-chosen shock-tail sides are OFF (the F4 risk-off tail stays, numbers-only).
USE_TEXT_TAIL = False
VOL_CAP_FAMILY = {"T2-F4": 1.2}   # these cards are already widened by the family prior
SCEN_MAX_SHIFT = 2.0   # per-scenario |shift| before recentring
N_SAMPLES = 3          # self-consistency samples (requests) per unit
MAX_REQUESTS = 8       # hard cap, well inside the 25/unit House allowance
REQ_TIMEOUT = 180       # per request (s)
TOTAL_BUDGET_S = 400     # whole text layer per unit (s); 71 units x 400 s stays well inside the 12 h stage clock


@dataclass
class LLMResult:
    adjust: Adjust
    status: str
    n_requests: int = 0
    raw: list = field(default_factory=list)
    per_asset: dict = field(default_factory=dict)
    scenarios: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    tail_sides: dict = field(default_factory=dict)   # asset -> -1 / 0 / +1 (majority of samples)
    tail_unanimous: dict = field(default_factory=dict)  # asset -> side only if every sample agrees
    event: bool = False                              # majority report a scheduled in-window event
    event_text: str = ""


SYSTEM = """You are a senior macro/rates/FX strategist producing a calibrated probabilistic forecast.
You stand on the AS-OF date. You may use ONLY the documents and numbers provided; do not use any
knowledge of what happened after the as-of date. If you recognise the episode, ignore what you
remember about the outcome - forecast only from what was knowable then.

A statistical engine has already produced a fat-tailed distribution for each target from the
price history (centre = current level, or the long-run drift for return targets; width = current
volatility regime, mean-reverting toward long-run volatility). Your job is ONLY to say how the
documents should modify that distribution. Most of the time the right answer is small or zero:
the engine is already well calibrated on average. Move things only when the text gives a reason.

Rules:
- direction: expected move of the target over the horizon RELATIVE to the engine centre, in units
  of the engine's horizon standard deviation (sd). Range -0.6..+0.6. Use 0 if unclear. A clearly
  telegraphed policy path not yet in prices might justify 0.2-0.4. Tone alone justifies at most 0.1-0.2.
- vol_mult: multiplier on the engine's width. 1.0 = keep. >1 when a scheduled binary event
  (vote, policy decision, data release, peg/floor test, crisis escalation) falls inside the window or
  the documents describe stress the recent prices do not yet show; <1 only if the documents
  ESTABLISH that a known source of variance is removed (an announced peg, a pre-committed path).
  Range 0.85..1.6.
- scenarios: only if the documents describe a genuinely branching outcome inside the window.
  Give 2-3 scenarios with probabilities summing to 1, each with a per-asset shift (in sd units,
  -2..2) and a vol_mult (0.6..1.8). Otherwise return an empty list.
- Mark each view "established" (the text states a decision/print/level) or "inferred" (tone).
- Mind the sign conventions: FX series are quoted as given in the unit (e.g. JPY = yen per USD, so
  yen strength is a FALL; EUR = USD per euro, so euro strength is a RISE). Yields in percent.
  Return targets are cumulative log returns of a long-short factor or the market.
Answer with a single JSON object and nothing after it."""

SCHEMA = """{
  "regime": "calm|normal|elevated|stressed|crisis",
  "event_in_window": true/false,
  "assets": {
    "<ASSET>": {"direction": <float>, "vol_mult": <float>, "basis": "established|inferred",
                "why": "<=30 words citing doc ids"}
  },
  "scenarios": [
    {"name": "...", "prob": <float>, "vol_mult": <float>, "shifts": {"<ASSET>": <float>}}
  ],
  "summary": "<=60 words"
}"""


def _panel_block(u, bb, extra: dict) -> str:
    lines = []
    for a in u.assets:
        inf = bb.info["assets"][a]
        ex = extra.get(a, {})
        cells = [(c, i) for i, c in enumerate(bb.cells) if c[0] == a]
        sds = ", ".join(f"h={c[1]}: centre {bb.mean[i]:.5g}, sd {bb.sd[i]:.4g}" for c, i in cells)
        lines.append(
            f"- {a}: last obs {inf['anchor']:.6g} on {inf['last_obs']}; "
            f"recent changes {ex.get('chg','')}; vol regime (current/long-run step sd) "
            f"{inf['sd_now']:.4g}/{inf['sd_lr']:.4g} = {inf['sd_now']/max(inf['sd_lr'],1e-12):.2f}x"
            f"{' [history STALE: target withheld for years before as-of]' if inf['stale'] else ''}; "
            f"engine: {sds}")
    return "\n".join(lines)


def _window_end(u) -> str:
    import pandas as pd
    try:
        if u.target_frequency == "monthly":
            per = (u.spec or {}).get("targets", {}).get("observation_periods")
            if per:
                return f"the {max(str(x)[:7] for x in per)} data release"
            return str((pd.Timestamp(u.asof) + pd.DateOffset(months=int(max(u.horizons) / 21) + 1)).date())
        return str((pd.Timestamp(u.asof) + pd.offsets.BDay(int(max(u.horizons)))).date())
    except Exception:
        return "the end of the horizon"


def build_prompt(u, bb, corpus_digest: str, extra: dict) -> str:
    unit = u.card.get("targets", {}).get("value_unit", "")
    hz = ", ".join(str(h) for h in u.horizons)
    return f"""AS-OF DATE: {u.asof}
CARD: {u.notes.get('title','')}
DESCRIPTION: {u.notes.get('description','')}
CARD NOTE ON THE TEXT: {u.notes.get('text_notes','')}
TARGET TYPE: {u.target_type} ({'cumulative log return over the horizon' if u.target_type=='log_return' else 'level on the target date'}); unit: {unit}
HORIZONS: {hz} ({'calendar months of the monthly series' if u.target_frequency=='monthly' else 'business days'} after the as-of)
FORECAST WINDOW: {u.asof} to about {_window_end(u)}

NUMERIC CONTEXT (from the panels, all <= as-of):
{_panel_block(u, bb, extra)}

DOCUMENTS (chronological, all dated <= as-of; long ones excerpted):
{corpus_digest}

Return JSON exactly in this shape (assets: {', '.join(u.assets)}):
{SCHEMA}"""


def _post(body: dict, timeout: int) -> dict:
    ep = os.environ["MODEL_ENDPOINT"].rstrip("/") + "/v1/chat/completions"
    req = urllib.request.Request(
        ep, data=json.dumps(body).encode(), method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ.get('MODEL_TOKEN', '')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:  # proxies come from the environment
        return json.loads(r.read().decode())


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = re.sub(r"```(?:json)?", "", text)
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for s in starts:
        depth = 0
        for i in range(s, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[s:i + 1])
                        if isinstance(obj, dict) and "assets" in obj:
                            return obj
                    except Exception:
                        pass
                    break
    return None


def _num(x, lo, hi, default):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(v):
        return default
    return min(max(v, lo), hi)


def query(u, bb, corpus_digest: str, extra: dict, log=print) -> LLMResult:
    if os.environ.get("T2_NO_LLM"):
        return LLMResult(Adjust(), "disabled in this build (T2_NO_LLM): numbers-only engine used")
    if not os.environ.get("MODEL_ENDPOINT"):
        return LLMResult(Adjust(), "offline: no MODEL_ENDPOINT (numbers-only engine used)")
    prompt = build_prompt(u, bb, corpus_digest, extra)
    model = os.environ.get("MODEL_NAME", "house")
    t0 = time.time()
    used = [0]
    raws: list = []

    def one(k: int, thinking: bool):
        left = TOTAL_BUDGET_S - (time.time() - t0)
        if used[0] >= MAX_REQUESTS or left < 30:
            return None
        used[0] += 1
        body = {"model": model, "temperature": 0.3, "top_p": 0.95, "seed": 1000 + k,
                "max_tokens": 4000,
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
                "chat_template_kwargs": {"enable_thinking": thinking}}
        try:
            resp = _post(body, int(min(REQ_TIMEOUT, left)))
            msg = resp["choices"][0]["message"]
            txt = msg.get("content") or ""
            obj = _extract_json(txt)
            if obj is None and msg.get("reasoning_content"):
                obj = _extract_json(msg["reasoning_content"])
            raws.append({"k": k, "thinking": thinking, "ok": obj is not None, "chars": len(txt)})
            return obj
        except Exception as e:  # network / HTTP / parse
            raws.append({"k": k, "thinking": thinking, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            return None

    with ThreadPoolExecutor(max_workers=N_SAMPLES) as ex:
        outs = list(ex.map(lambda k: one(k, True), range(N_SAMPLES)))
    good = [o for o in outs if o]
    # retry without thinking (cheaper, more reliable JSON) if too few parsed
    k = N_SAMPLES
    while len(good) < 2 and used[0] < MAX_REQUESTS and time.time() - t0 < TOTAL_BUDGET_S:
        o = one(k, False)
        k += 1
        if o:
            good.append(o)
        elif all("error" in r for r in raws[-2:]) and len(raws) >= 2:
            break
    if not good:
        return LLMResult(Adjust(), "fallback: no parsable House response (text-blind engine used)",
                         used[0], raws)
    return aggregate(u, good, used[0], raws)


def aggregate(u, objs: list[dict], n_req: int, raws: list) -> LLMResult:
    per_asset, adj = {}, Adjust()
    event = sum(bool(o.get("event_in_window")) for o in objs) > len(objs) / 2
    event_text = next((str(o.get("event", ""))[:200] for o in objs if o.get("event_in_window") and o.get("event")), "")
    vol_hi = VOL_HI_EVENT if event else VOL_HI
    shrink = VOL_SHRINK_EVENT if event else VOL_SHRINK
    fam_cap = VOL_CAP_FAMILY.get(u.family)
    tail_sides, tail_unan = {}, {}
    for a in u.assets:
        dirs, vols, basis, whys, sides = [], [], [], [], []
        for o in objs:
            r = (o.get("assets") or {}).get(a) or {}
            dirs.append(_num(r.get("direction"), -MAX_DIR_Z, MAX_DIR_Z, 0.0))
            vols.append(_num(r.get("vol_mult"), VOL_LO, vol_hi, 1.0))
            sd_ = str(r.get("tail_side", "both")).strip().lower()
            sides.append(-1 if sd_.startswith("down") else (1 if sd_.startswith("up") else 0))
            basis.append(str(r.get("basis", "inferred")))
            whys.append(str(r.get("why", ""))[:300])
        d_med = float(np.median(dirs))
        v_med = float(np.median(vols))
        established = sum(b.startswith("est") for b in basis) > len(basis) / 2
        v_app = math.exp(shrink * math.log(v_med))
        if not established:
            v_app = max(v_app, 1.0)   # inferred views may move the centre, never narrow
        if fam_cap:
            v_app = min(v_app, fam_cap)
        # tail side: needs a strict majority of samples naming the same side
        side = 0
        for sgn in (-1, 1):
            if sides.count(sgn) > len(sides) / 2:
                side = sgn
        tail_sides[a] = side
        tail_unan[a] = side if (len(sides) >= 2 and side != 0 and all(x == side for x in sides)) else 0
        per_asset[a] = {"direction_samples": dirs, "vol_samples": vols, "direction_applied":
                        DIR_SHRINK * d_med, "vol_applied": v_app,
                        "basis": "established" if established else "inferred", "why": whys,
                        "tail_side_samples": sides, "tail_side": side}
        adj.vol_mult[a] = v_app
    # scenarios: use only if a majority of samples report an in-window event with scenarios
    scen_sets = [o.get("scenarios") for o in objs if o.get("event_in_window") and o.get("scenarios")]
    scen_out = []
    if len(scen_sets) > len(objs) / 2:
        sc = max(scen_sets, key=lambda s: len(s) if isinstance(s, list) else 0)
        clean = []
        for s in sc[:3]:
            if not isinstance(s, dict):
                continue
            p = _num(s.get("prob"), 0.0, 1.0, 0.0)
            vm = _num(s.get("vol_mult"), 0.6, 1.8, 1.0)
            sh = {a: _num((s.get("shifts") or {}).get(a), -SCEN_MAX_SHIFT, SCEN_MAX_SHIFT, 0.0)
                  for a in u.assets}
            if p > 0:
                clean.append([p, sh, vm, str(s.get("name", ""))[:60]])
        if len(clean) >= 2:
            P = np.array([c[0] for c in clean])
            P = P / P.sum()
            for c, p in zip(clean, P):
                c[0] = float(p)
            # recentre: the mixture's mean shift is replaced by the shrunk directional view
            for a in u.assets:
                m = sum(c[0] * c[1][a] for c in clean)
                for c in clean:
                    c[1][a] = c[1][a] - m + per_asset[a]["direction_applied"]
            # never narrower than the engine x applied vol multiplier
            scale = {}
            for a in u.assets:
                tgt = per_asset[a]["vol_applied"] ** 2
                mbar = sum(c[0] * c[1][a] for c in clean)
                var = sum(c[0] * (c[2] ** 2 + (c[1][a] - mbar) ** 2) for c in clean)
                scale[a] = math.sqrt(tgt / var) if var < tgt else 1.0
            s_all = max(scale.values())
            for c in clean:
                c[2] *= s_all
            adj.scenarios = [(c[0], c[1], c[2]) for c in clean]
            adj.vol_mult = {}  # scenario vol multipliers carry the width now
            scen_out = clean
    if not adj.scenarios:
        adj.scenarios = [(1.0, {a: per_asset[a]["direction_applied"] for a in u.assets}, 1.0)]
    return LLMResult(adj, f"ok: {len(objs)} parsed House sample(s)", n_req, raws, per_asset, scen_out,
                     [str(o.get("summary", ""))[:400] for o in objs], tail_sides, tail_unan, event, event_text)
