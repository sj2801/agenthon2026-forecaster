# Research scripts

These scripts produced the backtest numbers in the main README. They were written for a research
workspace, so expect to adapt them before re-running:

- Run them from the repository root, e.g. `python research/embt.py`, with the organizers' public
  repository cloned next to this one as `../track2-forecasting-public`. Intermediate results go
  to `.cache/` (create it first).
- `finalbt.py` builds 2–12-day pseudo-cards; `shortbt*.py` build 1–4-day cards and test the FOMC
  event model; `embt.py` and `billbt.py` build the Monday-week EM FX, G10 and Treasury-bill cards;
  `ps1eval.py` scores the organizers' 22 short-horizon practice cards (it reads forecasts written
  beforehand by the CLI); `m0lib.py` implements the M0 baseline and the normalised composite;
  `events.py` / `events_research.py` collect FOMC dates for research.
- Some older scripts were first run against an earlier engine version, so re-running them can give
  slightly different numbers from those quoted in comments.
- Outcomes from the public data are used only for scoring here; nothing in this folder is copied
  into the Docker image.
