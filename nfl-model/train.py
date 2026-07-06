"""Train a first NFL win-probability model and grade it honestly.

Model: logistic regression on three pre-game features (elo_diff,
rest_diff, div_game). Deliberately simple — it's the baseline every
fancier model has to beat.

Split: train on 1999-2023, test on 2024-2025. The test seasons are
never touched during training, so the numbers below are an honest
preview of how the model would have done live.

Scoreboard, on the test seasons:
    accuracy    -- % of games picked correctly (home team baseline ~55%)
    Brier score -- error of the *probabilities*, lower is better.
                   Guessing 50% every game scores 0.250.
    calibration -- when the model says 65%, does that side win ~65%?
    vs market   -- same metrics for the vig-free probability implied by
                   the Vegas moneyline. This is the bar that matters.
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

from features import build_dataset

FEATURES = ["elo_diff", "rest_diff", "div_game"]
TEST_SEASONS = [2024, 2025]


def moneyline_to_prob(ml: pd.Series) -> pd.Series:
    """American moneyline -> raw implied probability (still includes vig)."""
    return (-ml / (-ml + 100)).where(ml < 0, 100 / (ml + 100))


def market_probs(df: pd.DataFrame) -> pd.Series:
    """Vig-free home win probability implied by the moneylines."""
    home = moneyline_to_prob(df["home_moneyline"])
    away = moneyline_to_prob(df["away_moneyline"])
    return home / (home + away)  # normalize so the two sides sum to 1


def report(name: str, y_true, y_prob):
    acc = accuracy_score(y_true, y_prob > 0.5)
    brier = brier_score_loss(y_true, y_prob)
    print(f"  {name:<22} accuracy {acc:.1%}   Brier {brier:.4f}")


def calibration_table(y_true: pd.Series, y_prob: pd.Series):
    bins = pd.cut(y_prob, [0, 0.4, 0.5, 0.6, 0.7, 1.0])
    table = pd.DataFrame({"predicted": y_prob, "actual": y_true}).groupby(
        bins, observed=True
    ).agg(games=("actual", "size"), predicted=("predicted", "mean"),
          actual=("actual", "mean"))
    print(table.to_string(float_format=lambda x: f"{x:.1%}"))


def main():
    df = build_dataset()
    train = df[~df["season"].isin(TEST_SEASONS)]
    test = df[df["season"].isin(TEST_SEASONS)].copy()
    print(f"Training on {len(train)} games ({train.season.min()}-{train.season.max()}), "
          f"testing on {len(test)} games ({TEST_SEASONS[0]}-{TEST_SEASONS[-1]})\n")

    model = LogisticRegression()
    model.fit(train[FEATURES], train["home_win"])

    coefs = dict(zip(FEATURES, model.coef_[0]))
    print("What the model learned (positive = helps the home team):")
    for name, c in coefs.items():
        print(f"  {name:<10} {c:+.4f}")
    print(f"  intercept  {model.intercept_[0]:+.4f}  <- home-field advantage\n")

    test["model_prob"] = model.predict_proba(test[FEATURES])[:, 1]

    print(f"Test results ({TEST_SEASONS[0]}-{TEST_SEASONS[-1]}):")
    report("always pick home team", test["home_win"], pd.Series(1.0, index=test.index))
    report("this model", test["home_win"], test["model_prob"])

    priced = test.dropna(subset=["home_moneyline", "away_moneyline"])
    report("Vegas moneyline", priced["home_win"], market_probs(priced))

    print("\nCalibration (model says X% -> home team actually wins Y%):")
    calibration_table(test["home_win"], test["model_prob"])

    out = test[["game_id", "season", "week", "home_team", "away_team",
                "model_prob", "home_win"]]
    out.to_csv("data/test_predictions.csv", index=False)
    print("\nWrote per-game predictions to data/test_predictions.csv")


if __name__ == "__main__":
    main()
