"""Build player-prop parlays: projections vs collected lines.

Reads the latest snapshot from the prop-line archive (props_collector),
scores every over/under against the projection models' probability
curves, and assembles parlays from the best legs.

The correlation rule — the part naive prop parlays get fatally wrong:
legs from the SAME game are correlated (a QB's passing yards and his
WR1's receiving yards move together; a blowout suppresses everyone's
volume). Multiplying their probabilities as if independent overstates
the parlay's chance. v1 takes the honest route: **max one leg per
game**, so independence is roughly true. Same-game parlays need an
explicit correlation model — a later project, not a shortcut.

Until the archive has data this prints guidance and exits; it comes
alive the week the collector starts writing.
"""
import argparse
import re

import pandas as pd

from props_collector import OUT as ARCHIVE
from props_features import build_market_dataset
from props_model import fit_quantile_models, predict_quantiles, prob_over

# The Odds API market key -> our modeled market
MARKET_MAP = {
    "player_receptions": "receptions",
    "player_reception_yds": "receiving_yards",
    "player_rush_yds": "rushing_yards",
    "player_pass_yds": "passing_yards",
}


def norm_name(s: str) -> str:
    s = re.sub(r"[^a-z ]", "", str(s).lower())
    return " ".join(s.split()[:2])  # first + last, punctuation-free


def american_decimal(price: float) -> float:
    return 1 + price / 100 if price > 0 else 1 + 100 / -price


def vig_free_over(over_price: float, under_price: float) -> float:
    po = 1 / american_decimal(over_price)
    pu = 1 / american_decimal(under_price)
    return po / (po + pu)


def latest_lines() -> pd.DataFrame:
    lines = pd.read_csv(ARCHIVE)
    lines = lines[lines["snapshot_utc"] == lines["snapshot_utc"].max()]
    lines = lines.dropna(subset=["line", "over_price", "under_price"])
    lines = lines[lines["market"].isin(MARKET_MAP)]
    lines["norm_player"] = lines["player"].map(norm_name)
    return lines


def score_legs(lines: pd.DataFrame) -> pd.DataFrame:
    """Attach model P(over) to every line we can match to a projection."""
    scored = []
    for api_market, market in MARKET_MAP.items():
        mlines = lines[lines["market"] == api_market]
        if mlines.empty:
            continue
        df, features = build_market_dataset(market)
        latest_season = df["season"].max()
        train = df[df["season"] <= latest_season]
        # Most recent feature row per player = current form
        current = (df.sort_values(["season", "week"])
                   .groupby("player_id").tail(1).copy())
        current["norm_player"] = current["player_display_name"].map(norm_name)

        models = fit_quantile_models(train, features, market)
        quants = predict_quantiles(models, current, features)

        merged = mlines.merge(
            current[["norm_player"]].assign(_idx=current.index),
            on="norm_player", how="inner")
        if merged.empty:
            continue
        q = quants.loc[merged["_idx"]].reset_index(drop=True)
        merged = merged.reset_index(drop=True)
        merged["p_over"] = prob_over(q, merged["line"])
        merged["market_p_over"] = merged.apply(
            lambda r: vig_free_over(r["over_price"], r["under_price"]), axis=1)
        scored.append(merged)
    if not scored:
        return pd.DataFrame()
    out = pd.concat(scored, ignore_index=True)

    # Choose the model's side of each line, and its edge vs the market
    over = out["p_over"] >= 0.5
    out["side"] = over.map({True: "over", False: "under"})
    out["p_model"] = out["p_over"].where(over, 1 - out["p_over"])
    out["p_market"] = out["market_p_over"].where(over, 1 - out["market_p_over"])
    out["price"] = out["over_price"].where(over, out["under_price"])
    out["edge"] = out["p_model"] - out["p_market"]
    return out


def build_parlays(legs: pd.DataFrame, k: int, top: int = 3) -> list[dict]:
    """Greedy: best remaining legs by model confidence, max one per game."""
    parlays = []
    pool = legs.sort_values("p_model", ascending=False)
    for _ in range(top):
        chosen, used_games = [], set()
        for row in pool.itertuples():
            if row.event_id in used_games or len(chosen) == k:
                continue
            chosen.append(row)
            used_games.add(row.event_id)
        if len(chosen) < k:
            break
        prob = payout = 1.0
        for leg in chosen:
            prob *= leg.p_model
            payout *= american_decimal(leg.price)
        parlays.append({
            "legs": [f"{c.player} {c.side} {c.line} {c.market}" for c in chosen],
            "prob": round(prob, 3), "payout": round(payout, 2),
            "ev": round(prob * payout - 1, 3),
        })
        pool = pool[~pool.index.isin([c.Index for c in chosen])]
    return parlays


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legs", type=int, default=3)
    args = ap.parse_args()

    if not ARCHIVE.exists():
        print("No prop-line archive yet. Once ODDS_API_KEY is configured and "
              "the collector workflow has run, parlays appear here.")
        return

    legs = score_legs(latest_lines())
    if legs.empty:
        print("No collected lines matched to projections yet.")
        return

    show = ["player", "market", "side", "line", "price", "p_model",
            "p_market", "edge"]
    print("Top legs by model confidence (max one per game in parlays):\n")
    print(legs.sort_values("p_model", ascending=False)[show]
          .head(12).to_string(index=False))
    print(f"\n{args.legs}-leg parlay candidates:")
    for p in build_parlays(legs, args.legs):
        print(f"  {' + '.join(p['legs'])}")
        print(f"    model {p['prob']:.0%} · pays {p['payout']}x · "
              f"EV {p['ev']:+.0%}")


if __name__ == "__main__":
    main()
