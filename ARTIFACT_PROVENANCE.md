# Artifact provenance — claude_final (Track 2 Final build)

## Executive summary (read this first)

The image contains code and a handful of scalar calibration parameters. It has no learned
model, no weights, no lookup tables and no task data. It calls no language model: the image is
built numbers-only and its descriptor lists `models: []`.

## Contents of the image

| Artifact | Source / version | Licence | Notes |
|---|---|---|---|
| `t2agent/` | written for this submission | MIT | numerical engine, corpus digest, House client, rationale writer |
| `qfbench2_track_forecasting/` | Agenthon-2026/track2-forecasting-public @ `60509df4` | MIT | used only for the official monthly observation-period step mapping |
| `qfbench2-common` | Agenthon2026-public tag v2.6.0 | per toolkit | installed at build time |
| numpy 2.1.3, pandas 2.2.3, pyarrow 18.1.0, jsonschema 4.23.0 | PyPI | BSD/Apache/MIT | pinned |

## Calibration parameters (`t2agent/params.py`)

- **Engine hyper-parameters.** These are the EWMA half-life, variance mean-reversion half-life,
  long window, correlation blend, drift shrinkage and the monthly settings. They were selected by
  coordinate descent on rolling-origin pseudo-cards cut from the public 2000–2024 panels (rates,
  G10 FX, JKP factors, macro monthly), scored on the official M0-normalized composite.
- **Final settings (claude_final).** These are Student-t ν = 6 and an overall width ×1.05. They
  were chosen on 6,212 unbiased pseudo-cards (UST yields and G10 FX, horizons 2–12 business days,
  origins every 15 business days 2003–2024; `research/finalbt.py`). Card-family width multipliers and
  the F4 shock tail are **switched off** (`fam_mult = {}`, `stress_tail = {}`), so no parameter
  is derived from practice-unit outcomes any more.
- **FOMC calendar (`t2agent/events.py`).** This is the public 2015–2026 schedule of FOMC
  decision days, known in advance, plus the March 2020 unscheduled decisions. It contains no
  market data or outcomes. The event-day variance ratios are measured at run time from the
  card's own panel history.
- All calibration data predate 2025. The image stores no unit identifier, date-specific value,
  target value or lookup table.

## House model

Not used by claude_final: `T2_NO_LLM=1` is set in the image, so the House client code that remains
in `t2agent/llm.py` is never reached, and the descriptor lists `models: []`.
