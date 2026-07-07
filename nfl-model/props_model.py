"""Player prop projection models with honest walk-forward backtesting.

For each market (receptions, receiving/rushing/passing yards) we train an
ensemble of quantile gradient-boosting models (P10/P25/P50/P75/P90).
That gives every player-week a probability *curve*, not a point guess —
"P(over 67.5 rushing yards)" falls straight out of it, which is exactly
the shape a prop bet asks for.

Backtest (per market, walk-forward on the last two completed seasons,
trained only on earlier seasons):

  MAE            -- median-projection error vs the naive baseline
                    (trailing 4-game average). Beating naive is table
                    stakes; report both.
  coverage       -- calibration of the quantiles: actual results should
                    land under the P10 line ~10% of the time, under the
                    P50 ~50%, etc. This is the number that must be right
                    for P(over) to mean anything.
  synthetic line -- Brier score of P(over) against a crude stand-in line
                    (the player's trailing average). Real prop lines
                    aren't public history; the collector is archiving
                    them from now on, and this backtest reruns against
                    the real archive as it accumulates.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from props_features import MARKETS, build_market_dataset

QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
N_TEST_SEASONS = 2


def fit_quantile_models(train: pd.DataFrame, features: list[str], market: str):
    models = {}
    for q in QUANTILES:
        m = HistGradientBoostingRegressor(
            loss="quantile", quantile=q, max_depth=3, learning_rate=0.06,
            max_iter=250, random_state=0)
        m.fit(train[features], train[market])
        models[q] = m
    return models


def predict_quantiles(models: dict, rows: pd.DataFrame,
                      features: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=rows.index)
    for q, m in models.items():
        out[q] = m.predict(rows[features])
    # Quantile crossing happens occasionally with independent models —
    # enforce monotonicity so the implied CDF is valid
    out = pd.DataFrame(np.sort(out.values, axis=1), index=out.index,
                       columns=QUANTILES).clip(lower=0)
    return out


def prob_over(quants: pd.DataFrame, line: pd.Series) -> pd.Series:
    """P(result > line) by interpolating the quantile curve, clamped to
    [0.02, 0.98] beyond the modeled tails."""
    qs = np.array(QUANTILES)
    probs = np.full(len(quants), np.nan)
    vals = quants.values
    for i in range(len(quants)):
        p_under = np.interp(line.iloc[i], vals[i], qs,
                            left=0.02, right=0.98)
        probs[i] = 1 - p_under
    return pd.Series(probs, index=quants.index)


def backtest_market(market: str) -> dict:
    df, features = build_market_dataset(market)
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-N_TEST_SEASONS:]

    rows = []
    for season in test_seasons:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        if train.empty or test.empty:
            continue
        models = fit_quantile_models(train, features, market)
        quants = predict_quantiles(models, test, features)

        naive = test[f"tr4_{market}"]
        line = naive  # synthetic stand-in until the real line archive grows
        p_over = prob_over(quants, line)
        over = (test[market] > line).astype(float)

        rows.append(pd.DataFrame({
            "actual": test[market], "p50": quants[0.50],
            "naive": naive, "p_over": p_over, "went_over": over,
            **{f"q{int(q*100)}": quants[q] for q in QUANTILES},
        }))

    r = pd.concat(rows)
    return {
        "market": market,
        "n": len(r),
        "mae_model": float((r["actual"] - r["p50"]).abs().mean()),
        "mae_naive": float((r["actual"] - r["naive"]).abs().mean()),
        "cov10": float((r["actual"] < r["q10"]).mean()),
        "cov50": float((r["actual"] < r["q50"]).mean()),
        "cov90": float((r["actual"] < r["q90"]).mean()),
        "brier_line": float(((r["p_over"] - r["went_over"]) ** 2).mean()),
        "brier_coinflip": float(((0.5 - r["went_over"]) ** 2).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", choices=list(MARKETS), default=None,
                    help="single market (default: all)")
    args = ap.parse_args()

    markets = [args.market] if args.market else list(MARKETS)
    results = [backtest_market(m) for m in markets]
    out = pd.DataFrame(results)
    out["mae_edge"] = 1 - out["mae_model"] / out["mae_naive"]
    with pd.option_context("display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))
    print("\ncov10/50/90 should read ~0.10/0.50/0.90 (quantile calibration)."
          "\nbrier_line vs brier_coinflip: how much the model knows about "
          "over/under a naive line.")


if __name__ == "__main__":
    main()
