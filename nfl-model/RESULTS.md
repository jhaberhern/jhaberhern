# Model Results

*Maintained automatically by `improve.py` — see the [workflow](../.github/workflows/improve-nfl-model.yml).*

## Current champion

- **Features:** elo_diff, rest_diff, div_game, spread_line
- **elo_k:** 24, **mov:** False, **model:** logreg, **C:** 1.0
- **Validation Brier** (seasons 2019-2023): 0.2125
- **Test Brier** (seasons 2024-2025): 0.2066 on 569 games (68.2% accuracy)
- **Vegas moneyline on the same test games:** 0.2059 — gap to close: +0.0007

## Run history

| date       |   games_in_data |   best_val_brier |   champion_val_brier |   champion_test_brier |   market_test_brier | new_champion   |
|:-----------|----------------:|-----------------:|---------------------:|----------------------:|--------------------:|:---------------|
| 2026-07-06 |            7261 |           0.2264 |               0.2264 |                0.2155 |              0.2059 | True           |
| 2026-07-06 |            7261 |           0.2239 |               0.2239 |                0.2143 |              0.2059 | True           |
| 2026-07-06 |            7261 |           0.2239 |               0.2239 |                0.2143 |              0.2059 | True           |
| 2026-07-06 |            7261 |           0.2231 |               0.2231 |                0.2142 |              0.2059 | True           |
| 2026-07-06 |            7261 |           0.2231 |               0.2231 |                0.2142 |              0.2059 | True           |
| 2026-07-07 |            7261 |           0.2231 |               0.2231 |                0.2142 |              0.2059 | True           |
| 2026-07-14 |            7261 |           0.2231 |               0.2231 |                0.2142 |              0.2059 | True           |
| 2026-07-17 |            7261 |           0.2125 |               0.2125 |                0.2066 |              0.2059 | True           |
| 2026-07-21 |            7261 |           0.2125 |               0.2125 |                0.2066 |              0.2059 | True           |
