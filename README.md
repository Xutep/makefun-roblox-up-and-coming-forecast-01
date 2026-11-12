# roblox-up-and-coming-forecast-01

Forecast Roblox's Up-and-Coming chart (device computer, country US), top 10,
place by place, for every hour of the 72 hours after entries close.

## How it works

`main.py` (Python 3.13 standard library only, no network) reads
`/input/history.ndjson` + `/input/task.json` and writes
`/output/forecast.json` (`probabilities` + `top10`).

Model: a Monte Carlo rank-diffusion forecast, seeded for determinism.

- Each game seen in the history gets a starting rank (its rank in the last
  snapshot, or just off the chart if it recently left it) and a drift
  (ranks per day) from its age: games first seen <1 day ago improve
  ~7 ranks/day, 1-2 days ~2/day, older games worsen ~4/day (they age
  toward the chart's 28-day limit). Two other signals were measured and
  dropped: recent rank velocity (drift autocorrelation is only ~0.2, and
  adding it hurt both backtests) and player-count growth (correlation
  with rank change ~-0.02, i.e. none).
- Each simulated hour, every game's rank is its drifted rank plus Gaussian
  noise with sd 5 ranks per sqrt(day), tuned on backtests (4-6 all score
  ~equally; the raw rank-change volatility is larger, but most of it is
  composition churn, not movement of the games that matter). Sorting the
  games each hour gives one simulated chart.
- The forecast probability of a game at a place is its share of simulated
  hours at that place, scaled so each place sums to 0.80, leaving 0.20
  unnamed: new games that appear during the window cannot be named, and
  the unnamed share is what they score.
- `top10` is ordered by total probability of being in the top 10.

Backtested on the practice history (train to 2026-11-04, score the next
10 reads; train to 2026-11-06, score the next 6): log score -39.57 vs
baseline -56.91 (+17.34) at the first cutoff and -31.17 vs -37.71 (+6.55) at
the second, scored with the exact rules (unnamed share, 0.0001 floor).
Parameters (sigma, age weight, velocity weight, per-place named sum)
were tuned on those two cutoffs; the chosen point sits on a flat
plateau (sigma 4-6 all score ~equally), not a sharp optimum.

## Run locally

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs twice byte-identically (fixed RNG seed).
