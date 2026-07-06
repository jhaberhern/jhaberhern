"""Build parlays from the champion model's win probabilities — and
backtest them honestly before trusting them.

Why parlays are a trap: the sportsbook's edge compounds per leg. One
-110 bet gives the book ~4.5%; a 3-leg parlay gives it ~13%. A "most
likely" parlay (stacking favorites) hits often, but the payout is priced
worse than the risk. The only escape is legs where the model's
probability genuinely beats the market's — so this script compares
three ways of picking legs:

    most_likely  -- the week's k most confident model picks
    market_favs  -- the week's k biggest favorites by moneyline
                    (what a casual bettor does; the control group)
    edge         -- the k legs where the model disagrees with the
                    market most, in the model's favor

Backtest mode replays every week of the held-out test seasons: train on
everything before them, pick a parlay each week, settle it at the real
moneyline payout, and report the ROI. Upcoming mode prints parlay picks
for scheduled games (payouts included once the book posts lines).

Paper trading only. Nothing here places bets, and nothing should.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from features import build_dataset
from improve import make_model
from train import market_probs

HERE = Path(__file__).parent
CHAMPION_FILE = HERE / "champion.json"

STRATEGIES = ("most_likely", "market_favs", "edge")


def load_champion() -> dict:
    return json.loads(CHAMPION_FILE.read_text())["config"]


def decimal_odds(ml: float) -> float:
    """American moneyline -> decimal payout multiplier (stake included)."""
    return 1 + ml / 100 if ml > 0 else 1 + 100 / -ml


def predict(config: dict, train: pd.DataFrame, games: pd.DataFrame) -> pd.Series:
    model = make_model(config)
    model.fit(train[config["features"]], train["home_win"])
    return pd.Series(model.predict_proba(games[config["features"]])[:, 1],
                     index=games.index)


def pick_sides(games: pd.DataFrame) -> pd.DataFrame:
    """For each game, work out the pick each strategy would make."""
    g = games.copy()
    g["market_prob"] = market_probs(g)  # vig-free home win probability

    # Each strategy picks a side and ranks by its own confidence measure
    g["ml_pick_home"] = g["model_prob"] >= 0.5
    g["ml_conf"] = g["model_prob"].where(g["ml_pick_home"], 1 - g["model_prob"])

    g["mf_pick_home"] = g["market_prob"] >= 0.5
    g["mf_conf"] = g["market_prob"].where(g["mf_pick_home"], 1 - g["market_prob"])

    g["edge_home"] = g["model_prob"] - g["market_prob"]
    g["edge_pick_home"] = g["edge_home"] > 0
    g["edge_conf"] = g["edge_home"].abs()
    return g


def leg_columns(strategy: str) -> tuple[str, str]:
    prefix = {"most_likely": "ml", "market_favs": "mf", "edge": "edge"}[strategy]
    return f"{prefix}_pick_home", f"{prefix}_conf"


def settle_parlay(legs: pd.DataFrame, pick_col: str) -> tuple[bool, float, float]:
    """Returns (hit, payout multiplier, model probability of the parlay)."""
    hit, payout, prob = True, 1.0, 1.0
    for leg in legs.itertuples():
        pick_home = getattr(leg, pick_col)
        hit &= bool(leg.home_win) == pick_home
        payout *= decimal_odds(leg.home_moneyline if pick_home
                               else leg.away_moneyline)
        prob *= leg.model_prob if pick_home else 1 - leg.model_prob
    return hit, payout, prob


def backtest(config: dict, n_test_seasons: int = 2, legs_options=(2, 3, 5)):
    df = build_dataset(elo_k=config["elo_k"], mov=config["mov"])
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[-n_test_seasons:]

    train = df[df["season"] < test_seasons[0]]
    test = df[df["season"].isin(test_seasons)].dropna(
        subset=["home_moneyline", "away_moneyline"]).copy()
    test["model_prob"] = predict(config, train, test)
    test = pick_sides(test)

    print(f"Backtest: one parlay per week, seasons {test_seasons[0]}-{test_seasons[-1]}, "
          f"1 unit staked each\n")
    rows = []
    for strategy in STRATEGIES:
        pick_col, conf_col = leg_columns(strategy)
        for k in legs_options:
            profits, hits, probs = [], 0, []
            for _, week in test.groupby(["season", "week"]):
                if len(week) < k:
                    continue  # not enough priced games (late playoff weeks)
                legs = week.nlargest(k, conf_col)
                hit, payout, prob = settle_parlay(legs, pick_col)
                profits.append(payout - 1 if hit else -1.0)
                hits += hit
                probs.append(prob)
            n = len(profits)
            rows.append({
                "strategy": strategy, "legs": k, "parlays": n,
                "hit_rate": f"{hits / n:.0%}",
                "model_expected": f"{pd.Series(probs).mean():.0%}",
                "profit_units": round(sum(profits), 1),
                "roi": f"{sum(profits) / n:+.0%}",
            })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nhit_rate = parlays that cashed; model_expected = what the model "
          "thought the hit rate would be;\nroi = profit per unit staked. "
          "Remember: the book's edge compounds per leg.")


def upcoming(config: dict, k: int):
    df = build_dataset(elo_k=config["elo_k"], mov=config["mov"],
                       include_upcoming=True)
    played = df[df["home_win"].notna()]
    future = df[df["home_win"].isna()]
    if future.empty:
        print("No upcoming games on the schedule.")
        return

    # Next scheduled week only — predictions further out degrade fast
    season = future["season"].min()
    week = future.loc[future["season"] == season, "week"].min()
    games = future[(future["season"] == season) & (future["week"] == week)].copy()
    games["model_prob"] = predict(config, played, games)
    games["pick"] = games["home_team"].where(games["model_prob"] >= 0.5,
                                             games["away_team"])
    games["conf"] = games["model_prob"].where(games["model_prob"] >= 0.5,
                                              1 - games["model_prob"])
    games = games.sort_values("conf", ascending=False)

    print(f"Season {season} week {week} — model picks, most confident first:\n")
    for g in games.itertuples():
        print(f"  {g.away_team:>3} @ {g.home_team:<3}  pick {g.pick:<3} "
              f"({g.conf:.0%})")

    legs = games.head(k)
    prob = legs["conf"].prod()
    print(f"\nMost likely {k}-leg parlay: "
          f"{' + '.join(legs['pick'])} — model gives it {prob:.0%}")
    if legs[["home_moneyline", "away_moneyline"]].notna().all().all():
        payout = 1.0
        for leg in legs.itertuples():
            payout *= decimal_odds(leg.home_moneyline if leg.model_prob >= 0.5
                                   else leg.away_moneyline)
        ev = prob * payout - 1
        print(f"Pays {payout:.2f}x at current lines — model EV {ev:+.0%} "
              f"per unit (paper only)")
    else:
        print("Lines not posted yet — payout and EV appear once the book "
              "prices the week.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backtest", action="store_true",
                    help="replay the test seasons week by week")
    ap.add_argument("--legs", type=int, default=3,
                    help="legs per parlay for upcoming mode")
    args = ap.parse_args()

    config = load_champion()
    print(f"Champion config: {config}\n")
    if args.backtest:
        backtest(config)
    else:
        upcoming(config, args.legs)


if __name__ == "__main__":
    main()
