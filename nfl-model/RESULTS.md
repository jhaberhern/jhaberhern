# Model Results

*Maintained automatically by `improve.py` — see the [workflow](../.github/workflows/improve-nfl-model.yml).*

## Current champion

- **Features:** elo_diff, rest_diff, div_game, margin_diff, home_qb_new, away_qb_new, qb_val_diff
- **elo_k:** 16, **mov:** True, **model:** logreg, **C:** 0.1
- **Validation Brier** (seasons 2019-2023): 0.2231
- **Test Brier** (seasons 2024-2025): 0.2142 on 569 games (65.9% accuracy)
- **Vegas moneyline on the same test games:** 0.2059 — gap to close: +0.0083

## Run history

| date       |   games_in_data |   best_val_brier |   champion_val_brier |   champion_test_brier |   market_test_brier | new_champion   |
|:-----------|----------------:|-----------------:|---------------------:|----------------------:|--------------------:|:---------------|
| 2026-07-06 |            7261 |           0.2264 |               0.2264 |                0.2155 |              0.2059 | True           |
| 2026-07-06 |            7261 |           0.2239 |               0.2239 |                0.2143 |              0.2059 | True           |
| 2026-07-06 |            7261 |           0.2239 |               0.2239 |                0.2143 |              0.2059 | True           |
| 2026-07-06 |            7261 |           0.2231 |               0.2231 |                0.2142 |              0.2059 | True           |
| 2026-07-06 |            7261 |           0.2231 |               0.2231 |                0.2142 |              0.2059 | True           |
