"""Automated model improvement: a champion-vs-challengers tournament.

Run weekly (by hand or via the GitHub Action), this script:

  1. Builds a grid of challenger configurations — different Elo settings,
     feature sets, and regularization strengths.
  2. Scores every challenger with walk-forward validation: for each
     validation season, train only on seasons before it, predict it,
     and average the Brier score. No peeking at the future, ever.
  3. Promotes a new champion only if it beats the reigning champion's
     validation score. The held-out test seasons are reported for
     context but NEVER used to pick the winner — that would be
     quietly overfitting to the test set.
  4. Logs every run to history.csv and rewrites RESULTS.md, so the
     repo carries a running record of the model getting better.

Seasons roll forward automatically: the two most recent seasons with
results are the test set, the five before that are validation.
"""
import itertools
import json
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss

from features import build_dataset
from train import market_probs

HERE = Path(__file__).parent
CHAMPION_FILE = HERE / "champion.json"
HISTORY_FILE = HERE / "history.csv"
RESULTS_FILE = HERE / "RESULTS.md"

N_TEST_SEASONS = 2
N_VAL_SEASONS = 5

# The search space. Small on purpose: every axis is here because it has a
# football reason to matter, and a small grid keeps validation meaningful.
ELO_K = [16, 20, 24]                      # how fast ratings react
FEATURE_SETS = [
    ("elo_diff", "rest_diff", "div_game"),
    ("elo_diff", "rest_diff", "div_game", "margin_diff"),
]
REG_C = [0.1, 1.0]                        # logistic regression regularization


def candidates():
    for k, feats, c in itertools.product(ELO_K, FEATURE_SETS, REG_C):
        yield {"elo_k": k, "features": list(feats), "C": c}


def walk_forward_brier(df: pd.DataFrame, cand: dict, val_seasons: list[int]) -> float:
    """Mean Brier score across validation seasons, each predicted using
    only the seasons that came before it."""
    scores = []
    for season in val_seasons:
        train = df[df["season"] < season]
        val = df[df["season"] == season]
        model = LogisticRegression(C=cand["C"])
        model.fit(train[cand["features"]], train["home_win"])
        prob = model.predict_proba(val[cand["features"]])[:, 1]
        scores.append(brier_score_loss(val["home_win"], prob))
    return float(pd.Series(scores).mean())


def evaluate_on_test(df: pd.DataFrame, cand: dict, test_seasons: list[int]) -> dict:
    train = df[~df["season"].isin(test_seasons)]
    test = df[df["season"].isin(test_seasons)]
    model = LogisticRegression(C=cand["C"])
    model.fit(train[cand["features"]], train["home_win"])
    prob = model.predict_proba(test[cand["features"]])[:, 1]

    priced = test.dropna(subset=["home_moneyline", "away_moneyline"])
    return {
        "test_brier": round(brier_score_loss(test["home_win"], prob), 4),
        "test_accuracy": round(accuracy_score(test["home_win"], prob > 0.5), 4),
        "market_brier": round(
            brier_score_loss(priced["home_win"], market_probs(priced)), 4),
        "n_test_games": len(test),
    }


def load_champion() -> dict | None:
    if CHAMPION_FILE.exists():
        return json.loads(CHAMPION_FILE.read_text())
    return None


def write_results(champion: dict, history: pd.DataFrame):
    gap = champion["test_brier"] - champion["market_brier"]
    lines = [
        "# Model Results",
        "",
        "*Maintained automatically by `improve.py` — see the "
        "[workflow](../.github/workflows/improve-nfl-model.yml).*",
        "",
        "## Current champion",
        "",
        f"- **Features:** {', '.join(champion['config']['features'])}",
        f"- **Elo K:** {champion['config']['elo_k']}, "
        f"**regularization C:** {champion['config']['C']}",
        f"- **Validation Brier** (seasons {champion['val_seasons']}): "
        f"{champion['val_brier']:.4f}",
        f"- **Test Brier** (seasons {champion['test_seasons']}): "
        f"{champion['test_brier']:.4f} on {champion['n_test_games']} games "
        f"({champion['test_accuracy']:.1%} accuracy)",
        f"- **Vegas moneyline on the same test games:** "
        f"{champion['market_brier']:.4f} — gap to close: {gap:+.4f}",
        "",
        "## Run history",
        "",
        history.tail(20).to_markdown(index=False),
        "",
    ]
    RESULTS_FILE.write_text("\n".join(lines))


def main():
    df = build_dataset()  # elo variants rebuilt per candidate below
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-N_TEST_SEASONS:]
    val_seasons = seasons[-(N_TEST_SEASONS + N_VAL_SEASONS):-N_TEST_SEASONS]
    print(f"{len(df)} games | validation {val_seasons} | test {test_seasons}\n")

    datasets = {k: build_dataset(elo_k=k) for k in ELO_K}

    results = []
    for cand in candidates():
        data = datasets[cand["elo_k"]]
        # Validation must not include test seasons
        val_data = data[~data["season"].isin(test_seasons)]
        brier = walk_forward_brier(val_data, cand, val_seasons)
        results.append((brier, cand))
        print(f"  val Brier {brier:.4f}  {cand}")

    best_brier, best = min(results, key=lambda r: r[0])

    champion = load_champion()
    promoted = champion is None or best_brier < champion["val_brier"] - 1e-6
    if promoted:
        test_metrics = evaluate_on_test(datasets[best["elo_k"]], best, test_seasons)
        champion = {
            "config": best,
            "val_brier": round(best_brier, 4),
            "val_seasons": f"{val_seasons[0]}-{val_seasons[-1]}",
            "test_seasons": f"{test_seasons[0]}-{test_seasons[-1]}",
            **test_metrics,
            "promoted_on": date.today().isoformat(),
        }
        CHAMPION_FILE.write_text(json.dumps(champion, indent=2) + "\n")
        print(f"\nNEW CHAMPION (val Brier {best_brier:.4f}): {best}")
    else:
        print(f"\nChampion defends its title "
              f"(best challenger {best_brier:.4f} vs champion "
              f"{champion['val_brier']:.4f})")

    row = pd.DataFrame([{
        "date": date.today().isoformat(),
        "games_in_data": len(df),
        "best_val_brier": round(best_brier, 4),
        "champion_val_brier": champion["val_brier"],
        "champion_test_brier": champion["test_brier"],
        "market_test_brier": champion["market_brier"],
        "new_champion": promoted,
    }])
    history = (pd.concat([pd.read_csv(HISTORY_FILE), row])
               if HISTORY_FILE.exists() else row)
    history.to_csv(HISTORY_FILE, index=False)
    write_results(champion, history)
    print(f"Updated {HISTORY_FILE.name} and {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
