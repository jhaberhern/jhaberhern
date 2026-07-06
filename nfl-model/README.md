# NFL Win Probability Model

Training my own AI to predict NFL games — and grading it against the
toughest opponent there is: the Vegas betting market.

## The idea

A sportsbook's odds are already a very good prediction model. At standard
-110 odds you need to be right **52.4%** of the time just to break even,
so the realistic goal isn't "beat Vegas" on day one — it's to build a
model, measure it *honestly*, and improve it one feature at a time.
Ground rules: paper-trade only (track hypothetical picks, no money), and
never automate actual bet placement.

## Quick start

```bash
pip install -r requirements.txt
python3 download_data.py   # every NFL game since 1999, via nflverse
python3 train.py           # train, evaluate, write test predictions
```

## How it works

| File | What it does |
|------|-------------|
| `download_data.py` | Fetches the [nflverse](https://github.com/nflverse/nfldata) games dataset: scores, rest days, and Vegas lines for every game since 1999 |
| `features.py` | Builds pre-game features: **Elo ratings** (computed chronologically so there's no data leakage), rest-day difference, divisional-game flag |
| `train.py` | Trains a logistic regression on 1999–2023, tests on 2024–2025, and compares against two baselines: "always pick the home team" and the vig-free probability implied by the Vegas moneyline |

## First results (test seasons 2024–2025, never seen in training)

| Predictor | Accuracy | Brier score* |
|-----------|---------:|------------:|
| Always pick home team | 54.1% | 0.4587 |
| **This model** | **65.6%** | **0.2198** |
| Vegas moneyline | 68.4% | 0.2059 |

*Brier score grades the probabilities, not just the picks — lower is
better, and guessing 50% every game scores 0.250.

A three-feature model gets ~two-thirds of games right and lands within a
few points of the market. The gap that's left (0.2198 vs 0.2059) is the
mountain: it's where injuries, quarterbacks, weather, and matchups live.

Calibration is honest too — when the model says 65%, that side wins about
65% of the time. The one soft spot is the 40–50% bucket (toss-up games it
slightly mis-ranks), which is a good first thing to investigate.

## Ideas for next steps

- **QB adjustment** — the single biggest thing Elo misses. A backup QB
  starting is worth several points; the dataset has `home_qb_name` /
  `away_qb_name` ready to use.
- **Point-differential features** — rolling average margin over the last
  N games catches form that win/loss Elo smooths over.
- **Compare picks to the market weekly** — flag games where the model
  disagrees with the moneyline by enough to clear the vig, log them, and
  track results over a season (paper only).
- **Closing line value** — the metric pros use: do the model's picks beat
  the closing line, even before results come in?
- **Auto-retrain** — a GitHub Action that re-downloads data weekly during
  the season and updates this README with the model's running record.
