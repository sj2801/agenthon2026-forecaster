"""Hyper-parameters for the Final build (claude-final).

Engine settings (EWMA, mean reversion, windows, correlation, monthly settings) were selected by
rolling-origin backtests on the 2000-2024 public panels and are unchanged from claude2-nollm.

Final-specific changes (2026-10-11), chosen on 6,212 unbiased pseudo-cards that match the ranked
Final (US Treasury yields + G10 FX, horizons 2-12 business days, origins every 15 business days
2003-2024, official M0-normalized composite; see research/finalbt.py):

* fam_mult = {} and stress_tail = {}: the card label (F1-F4) no longer changes the width and no
  F4 shock tail is added. Those priors were fitted to hindsight-selected practice cards; on
  unbiased cards an F2 label (x1.75) scored 1.20 and an F4 label (x2.5 + tail) 1.88, against
  1.016 for the plain engine and 1.056 for M0.
* nu 5 -> 6 and vol_mult 1.0 -> 1.05: slightly thinner tails with a 5% scheduled-event width
  allowance. Every ranked window spans an FOMC decision; on windows containing an FOMC decision
  (10 complete-calendar years) this scores 1.040 vs 1.049, and elsewhere it is unchanged
  (0.992 vs 0.991).

Update 2026-10-13 (ranked Final now: as-of 26 Oct, horizons 1-4 business days, FOMC decision on
28 Oct inside every window):

* event_shrink = 0.75 switches on the FOMC event-day variance model (engine._event_weights,
  calendar in events.py). Each series' own history gives the variance ratio of the step that
  contains a decision and of the first step after it; the forecast step holding the 28 Oct
  decision gets that extra variance (shrunk 25% toward 1), other steps slightly less so the average
  is unchanged. Out of sample (2,866 one-to-four-day UST/FX cards with origins 2017-07..2024,
  estimates from data before each origin): windows containing a decision 1.457 -> 1.424 (M0 1.566),
  other windows unchanged (1.039), all cards 1.083 -> 1.080 (M0 1.124). See research/shortbt*.py.

Update 2026-10-21 (short-horizon practice set + 9,486 Monday-week backtest cards, research/embt.py):

* Treasury-bill cards (every asset a UST yield of maturity <= 2 years, at least one bill) use
  SHORT_UST: nu 4 and no extra width. Bills move in small steps with occasional jumps. On 3,312
  Monday-to-Friday bill cards this scored 1.183 vs 1.218 (M0 1.325), better in 2003-12, 2013-22
  and FOMC weeks alike, and on the practice set's 8 rate cards 1.14 vs 1.18.
* EM FX (ECB crosses) and G10 keep the settings above: 0.995 vs M0 1.049 on ordinary EM weeks,
  1.122 vs 1.172 on FOMC weeks.
"""

TUNED: dict = {
    "ewma_hl": 20.0,
    "vol_hl": 150.0,
    "lr_window": 2500,
    "nu": 6.0,
    "vol_mult": 1.05,
    "drift_logret": 0.0,
    "drift_level": 0.0,
    "corr_shrink": 0.25,
    "corr_hl": 60.0,
    "m_ewma_hl": 3.0,
    "m_vol_hl": 12.0,
    "m_drift_window": 6,
    "m_drift_shrink": 1.0,
    "m_lr_window": 240,
    "fam_mult": {},
    "stress_tail": {},
    "event_shrink": 0.75,
}

# Treasury-bill cards: fatter tails, no extra width (see the docstring).
SHORT_UST: dict = {"nu": 4.0, "vol_mult": 1.0}


def card_overrides(assets) -> dict:
    """Setting overrides that depend only on which series a card asks for."""
    import re

    mats = []
    for a in assets:
        m = re.fullmatch(r"UST_(\d+)([MY])", str(a))
        if not m:
            return {}
        mats.append(int(m.group(1)) / (12.0 if m.group(2) == "M" else 1.0))
    if mats and min(mats) < 2 and max(mats) <= 2:
        return dict(SHORT_UST)
    return {}
