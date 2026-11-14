# Probabilistic FX & Treasury forecaster — Agenthon 2026, Track 2

A containerised agent that produces **joint probabilistic forecasts** (4,000 scenario draws) for US
Treasury yields and FX rates over 1–4 business days. It was built for **Track 2 (Time-Series
Forecasting) of Agenthon 2026**, a NeurIPS 2026 competition for finance agents.

Forecasts are scored with a proper scoring rule: 50% CRPS, 30% variogram score (joint structure
across assets and horizons) and 20% tail pinball loss. Each part is normalised by the organizers'
baseline **M0**, a Gaussian random walk, and lower scores are better.

> **Final result: rank 23 of 83 scored teams** (team *Pawsibble*, composite 1.0011). The Final
> was ranked on live market outcomes for 27–30 Oct 2026 (FOMC week); the winning score was 0.9467.

## Results

**Historical backtest: 10,728 weekly cards.** Each card has a Monday as-of date and targets
Tuesday to Friday, mirroring the Final design.

| Series | Cards | Years | Model | M0 | Better than M0 | Weeks won |
|---|---|---|---|---|---|---|
| EM FX (MXN, BRL, ZAR, KRW, INR) | 3,402 | 2011–2021 | 1.006 | 1.060 | 5.1% | 60% |
| G10 FX | 2,772 | 2003–2024 | 1.046 | 1.085 | 3.6% | 56% |
| UST 2Y–30Y | 1,242 | 2017–2024 | 1.106 | 1.160 | 4.6% | 52% |
| UST bills 3M/6M/1Y | 3,312 | 2003–2022 | 1.183 | 1.325 | 10.7% | 65% |
| **All** | **10,728** | | **1.083** | **1.160** | **6.6%** | **60%** |

The model beats M0 in 55 of 61 series-years. The largest gains come in regime shifts (2008,
2020–21), and the model is also better in every series during FOMC weeks.

**Organizers' short-horizon practice set: 22 cards with published outcomes.**

| | Composite | CRPS | Variogram | Tail pinball |
|---|---|---|---|---|
| This model | **1.21** | 0.98 | 1.69 | 1.07 |
| M0 baseline | 2.25 | 1.22 | 5.20 | 1.63 |

The model's 90% forecast intervals contained 47 of 53 realised outcomes (89%).

*Some settings were chosen on parts of the same history, so out-of-sample gains may be somewhat
smaller. The FOMC effect is estimated only from data available before each test week.*

## Method

1. **Increments.** Level series are modelled in daily changes; return series in log-returns.
2. **Volatility term structure.** An EWMA "current regime" variance (20-day half-life) decays
   toward a long-run variance (10-year window) with a 150-day half-life.
3. **Correlation.** Correlations blend 25% long-run and 75% EWMA, then are repaired to a valid
   correlation matrix.
4. **Exact joint path covariance.** One covariance covers every asset and horizon, so a 4-day
   forecast shares the 2-day path with the 2-day forecast.
5. **Fat tails.** Draws use a multivariate Student-t scale mixture (ν = 6, variance preserved)
   with antithetic pairs.
6. **Scheduled-event variance.** For the FOMC decision inside the forecast window, each series'
   own history measures how much more the decision step moves than an ordinary step. About 1.9×
   for 2-year yields on the day, and about 2.3× for FX on the next fixing, because fixings are
   taken before the 2 pm statement. That extra variance is placed on the matching step.
7. **Treasury-bill rule.** Cards made only of short Treasury yields use fatter tails (ν = 4).
8. **Safety net.** If anything fails on an unfamiliar card format, the organizers' reference
   forecaster writes a valid baseline forecast, so no card is lost.

The image calls no language model and contains no task data, answers or lookup tables; see
[`ARTIFACT_PROVENANCE.md`](ARTIFACT_PROVENANCE.md).

### What did not help

These ideas were tested and dropped because they did not improve both validation splits and the
historical backtest:

- filtered historical simulation;
- variance-ratio horizon scaling;
- regime ensembles;
- PCA, Ledoit–Wolf and Kendall correlation estimators;
- more draws;
- a text-uncertainty index;
- an LLM text layer, which added roughly zero on the development board.

## Run it

```bash
docker build --platform linux/amd64 -t agenthon-t2:final .
docker run --rm --network=none \
  -v /path/to/unit:/input:ro -v "$PWD/out":/output \
  agenthon-t2:final forecast --panels /input/panels/ --text /input/text/ \
  --asof 2016-06-17 --out /output/forecast.parquet
```

The run writes `forecast.parquet` (long format: draw, asset, horizon, value),
`forecast_meta.json` and `forecast_rationale.md`. Practice units are in the organizers' public
repository:
[Agenthon-2026/track2-forecasting-public](https://github.com/Agenthon-2026/track2-forecasting-public).

## Repository layout

| Path | Contents |
|---|---|
| `t2agent/` | The agent: data loading, engine, event calendar, rationale writer, CLI |
| `qfbench2_track_forecasting/` | The organizers' MIT reference package, unmodified; used for the monthly step mapping and the safety net |
| `research/` | Backtest and evaluation scripts; see [`research/README.md`](research/README.md) |
| `ARTIFACT_PROVENANCE.md` | What is inside the image and where every calibrated number came from |

## Credits

- Built by Qian (UTAR) for Agenthon 2026, Track 2.
- Developed with the help of an AI assistant (Claude), used for research, coding and
  backtesting under my direction.
- `qfbench2_track_forecasting/` is © the Agenthon 2026 organizers, MIT licensed; see its
  `LICENSE-track2`.
