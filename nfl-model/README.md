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
| `improve.py` | The self-improvement loop: a champion-vs-challengers tournament over Elo settings, feature sets, and regularization, scored by walk-forward validation. Promotes a new champion only when it wins on validation seasons; logs every run to `history.csv` and `RESULTS.md` |
| `parlay.py` | Builds parlays from the champion's win probabilities. `--backtest` replays the test seasons week by week and settles each parlay at real moneyline payouts; default mode prints picks (and EV, once lines are posted) for the next scheduled week |
| `ledger.py` | The paper-trade ledger: logs the champion's pick for every priced game of the upcoming week, then grades each pick against the result *and the closing line* (CLV). Season-to-date record lives in [`LEDGER.md`](LEDGER.md) |
| `export_dashboard.py` | Bundles champion metrics, history, picks, parlay, and ledger record into `docs/data.json` for the GitHub Pages dashboard (`docs/index.html`) |

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

## Getting 0.1% better every week, automatically

A [GitHub Action](../.github/workflows/improve-nfl-model.yml) runs every
Tuesday (after Monday Night Football): it downloads fresh data, runs the
`improve.py` tournament, and commits the results. The current champion
and full run history live in [`RESULTS.md`](RESULTS.md).

The improvement loop is built to be honest with itself:

- Challengers are scored with **walk-forward validation** — each
  validation season is predicted using only seasons that came before it.
- A challenger is promoted only if it beats the reigning champion on
  validation. The **test seasons are never used to pick winners** — they
  exist only to report how the champion would have done live.
- Splits roll forward on their own: the two most recent seasons are
  always the test set, the five before that are validation.

First tournament result: adding a rolling point-margin feature
(`margin_diff`) beat the original model — test Brier improved from
0.2198 to 0.2155 (66.1% accuracy).

The tournament's challenger axes so far: Elo speed (K), margin-of-victory
Elo updates, rolling point-margin windows (3/5/10 games), QB-change
flags, regularization strength, and gradient boosting vs logistic
regression. The current best blend lives in `champion.json`.

## The parlay question

The end goal is picking parlays, so `parlay.py` faces the math head-on:
the book's edge **compounds per leg** (~4.5% on one -110 bet, ~13% on a
3-leg parlay). Backtesting one parlay per week over 2024–2025, three
ways of choosing legs:

| Strategy | 3-leg hit rate | 3-leg ROI |
|----------|---------------:|----------:|
| Model's most confident picks | 57% | +14% |
| Biggest market favorites (control) | 57% | +1% |
| Biggest model-vs-market disagreements | 2% | −85% |

Three honest lessons in that table:

1. **~40 parlays is a tiny sample.** The positive ROI rows are luck-
   assisted: favorites hit well above the model's own expected rates in
   these seasons (57% actual vs 49% expected), and the model itself
   would predict thinner margins long-run. Do not read +14% as an edge.
2. **"Most likely parlay" ≠ profitable parlay.** Stacking favorites
   maximizes hit rate, but the payout is priced for it.
3. **Chasing disagreements with the market is fatal.** When a model
   that's weaker than the market (Brier 0.214 vs 0.206) disagrees with
   it, the model is usually the one that's wrong — and parlays multiply
   that mistake. The −85% row is the tuition bill.

The constructive takeaway: the route to parlays worth taking is a model
that beats the market *per game* first. Until then, this stays paper.

## The bankroll gates

The long-term goal is a model that pays for itself. That claim gets
earned through gates, not vibes:

1. **Gate 1 — beat the market on paper:** champion Brier better than the
   closing line's, or demonstrably positive closing line value (CLV).
   *Currently below this gate.*
2. **Gate 2 — a full season of receipts:** `ledger.py` logs every pick
   in advance and grades it against the result and the closing line.
   Sustained positive CLV over a season is the only pass mark — a
   winning record without CLV is variance wearing a suit.
3. **Gate 3 — minimum stakes,** sized by Kelly, only after Gate 2, and
   the stakes never grow faster than the evidence.

## Ideas for the next challenger

- **QB quality, not just QB change** — the flags catch a new starter but
  not whether he's Mahomes or a practice-squad guy. A per-QB rating
  (games started, or a QB-level Elo) is the natural upgrade.
- **Compare picks to the market weekly** — flag games where the model
  disagrees with the moneyline by enough to clear the vig, log them, and
  track results over a season (paper only).
- **Closing line value** — the metric pros use: do the model's picks beat
  the closing line, even before results come in?
- **Situational features** — weather (wind), short-week road games,
  post-bye rest, West Coast teams in early East Coast kickoffs.
