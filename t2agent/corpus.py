"""Corpus reading with the cutoff enforced locally, plus a relevance-ranked digest for the LLM.

Nothing dated after --asof is ever opened.  Long documents (Beige Book, minutes) are reduced to
their most relevant paragraphs with a transparent keyword score, so the House model sees the
material that bears on the target assets inside a bounded prompt.
"""

from __future__ import annotations

import json
import pathlib
import re
from dataclasses import dataclass

ASSET_TERMS = {
    "UST": ["treasury", "yield", "rates", "funds rate", "federal funds", "policy rate", "tighten",
            "hike", "cut", "easing", "balance sheet", "inflation", "term premium", "issuance",
            "duration", "curve", "basis points", "accommodat", "restrictive", "dot", "projection"],
    "EUR": ["euro", "ecb", "eurozone", "euro area", "governing council", "deposit facility"],
    "GBP": ["sterling", "pound", "bank of england", "mpc", "referendum", "brexit", "united kingdom", "uk "],
    "JPY": ["yen", "bank of japan", "boj", "japan", "yield curve control", "negative interest", "jgb"],
    "CHF": ["franc", "swiss", "snb", "safe haven", "floor"],
    "AUD": ["australia", "rba", "iron ore", "commodity", "china"],
    "NZD": ["new zealand", "rbnz", "dairy", "commodity"],
    "CAD": ["canada", "bank of canada", "oil", "crude", "energy"],
    "NOK": ["norway", "norges", "oil", "crude", "energy"],
    "SEK": ["sweden", "riksbank"],
    "DKK": ["denmark", "danmarks", "peg", "erm"],
    "CNY": ["china", "renminbi", "yuan", "pboc", "tariff", "trade"],
    "INR": ["india", "rupee", "emerging", "capital outflow", "taper"],
    "BRL": ["brazil", "real", "emerging", "capital outflow", "taper"],
    "MKT": ["equity", "stock", "market", "financial conditions", "volatility", "earnings", "risk"],
    "SMB": ["small", "equity", "credit"], "HML": ["value", "bank", "rates", "equity"],
    "MOM": ["momentum", "reversal", "equity", "volatility"], "QMJ": ["quality", "equity", "credit"],
    "BAB": ["leverage", "equity", "beta"],
    "CPI": ["inflation", "consumer price", "cpi", "prices", "shelter", "energy", "core"],
    "UNRATE": ["unemployment", "labor market", "payroll", "jobless", "hiring", "layoff"],
    "NFP": ["payroll", "employment", "labor market", "jobs", "hiring", "layoff"],
    "PCE": ["inflation", "pce", "prices"],
}
GENERIC = ["outlook", "risk", "uncertain", "expect", "will", "likely", "downside", "upside",
           "stress", "shock", "crisis", "turmoil", "surprise", "pandemic", "virus", "war",
           "default", "downgrade", "debt ceiling", "election", "vote", "intervention"]


@dataclass
class Doc:
    doc_id: str
    timestamp: str
    doc_type: str
    source: str
    text: str


def read_corpus(text_dir: pathlib.Path | None, asof: str) -> tuple[list[Doc], int]:
    if text_dir is None or not text_dir.is_dir():
        return [], 0
    idx_path = text_dir / "corpus_index.json"
    docs, excluded = [], 0
    entries = []
    if idx_path.exists():
        try:
            entries = json.loads(idx_path.read_text()).get("documents", [])
        except Exception:
            entries = []
    seen = set()
    for e in entries:
        ts = str(e.get("timestamp", ""))[:10]
        f = text_dir / str(e.get("file", ""))
        if not ts or ts > asof:
            excluded += 1
            continue
        if not f.is_file() or f.resolve().parent != text_dir.resolve():
            continue
        seen.add(f.name)
        docs.append(Doc(str(e.get("doc_id", f.stem)), ts, str(e.get("doc_type", "")),
                        str(e.get("source", "")), f.read_text(errors="replace")))
    # documents not in the index: only if a date in the filename is <= asof
    for f in sorted(text_dir.glob("*.txt")):
        if f.name in seen:
            continue
        m = re.findall(r"(\d{4})-?(\d{2})-?(\d{2})", f.name)
        if not m:
            excluded += 1
            continue
        ts = "-".join(m[-1])
        if ts > asof:
            excluded += 1
            continue
        docs.append(Doc(f.stem, ts, "unindexed", "", f.read_text(errors="replace")))
    docs.sort(key=lambda d: d.timestamp)
    return docs, excluded


def _terms(assets: list[str], notes: str) -> list[str]:
    t = list(GENERIC)
    for a in assets:
        for k, v in ASSET_TERMS.items():
            if a.startswith(k) or k.startswith(a):
                t += v
    words = re.findall(r"[a-zA-Z]{5,}", notes.lower())
    t += list(dict.fromkeys(words))[:40]
    return list(dict.fromkeys(x.lower() for x in t))


def digest(docs: list[Doc], assets: list[str], notes: str, budget_chars: int = 48000,
           per_doc_min: int = 1500) -> str:
    """Chronological digest. Short docs whole; long docs cut to their highest-scoring paragraphs."""
    if not docs:
        return "(no documents)"
    terms = _terms(assets, notes)
    n = len(docs)
    # later documents get more room: they carry the freshest information at the as-of
    w = [1.0 + 1.0 * i / max(n - 1, 1) for i in range(n)]
    caps = [max(per_doc_min, int(budget_chars * wi / sum(w))) for wi in w]
    parts = []
    for d, cap in zip(docs, caps):
        body = re.sub(r"[ \t]+", " ", d.text).strip()
        if len(body) > cap:
            paras = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[A-Z])", body) if len(p.strip()) > 60]
            scored = []
            for i, p in enumerate(paras):
                pl = p.lower()
                s = sum(pl.count(k) for k in terms) / (1 + len(p) / 800)
                if i < 2:
                    s += 2.0  # lead paragraphs usually state the decision / headline
                scored.append((s, i, p))
            keep, used = [], 0
            for s, i, p in sorted(scored, key=lambda x: -x[0]):
                if used + len(p) > cap:
                    if used == 0:
                        keep.append((i, p[:cap]))
                        used = cap
                    continue
                keep.append((i, p))
                used += len(p)
            body = "\n...\n".join(p for _, p in sorted(keep))
        parts.append(f"### [{d.doc_id}] {d.timestamp} | {d.doc_type} | {d.source}\n{body}")
    return "\n\n".join(parts)
