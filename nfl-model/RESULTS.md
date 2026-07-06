# Model Results

*Maintained automatically by `improve.py` — see the [workflow](../.github/workflows/improve-nfl-model.yml).*

## Current champion

- **Features:** elo_diff, rest_diff, div_game, margin_diff
- **Elo K:** 24, **regularization C:** 0.1
- **Validation Brier** (seasons 2019-2023): 0.2264
- **Test Brier** (seasons 2024-2025): 0.2155 on 569 games (66.1% accuracy)
- **Vegas moneyline on the same test games:** 0.2059 — gap to close: +0.0096

## Run history

| date       |   games_in_data |   best_val_brier |   champion_val_brier |   champion_test_brier |   market_test_brier | new_champion   |
|:-----------|----------------:|-----------------:|---------------------:|----------------------:|--------------------:|:---------------|
| 2026-07-06 |            7261 |           0.2264 |               0.2264 |                0.2155 |              0.2059 | True           |
