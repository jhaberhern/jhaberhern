"""Paper-trade ledger with closing line value (CLV) tracking.

This is Gate 2 of the bankroll plan: before any model earns real money,
it has to log a full season of picks *in advance* and prove two things —
that the picks make money at the odds available when logged, and that
the closing lines move toward them (positive CLV). CLV is the metric
professionals trust: bettors who consistently beat the closing line
profit long-run; bettors who don't, don't, even when they're running hot.

Run weekly (the GitHub Action does this every Tuesday), it:

  1. Grades pending picks whose games have since been played: result,
     profit at the logged odds (1 unit stakes), and CLV — how far the
     vig-free closing probability moved toward (or away from) our side
     between logging and kickoff.
  2. Logs the champion model's pick for every game in the next
     scheduled week that has a moneyline posted. Games priced later get
     logged by a later run; nothing is ever logged twice or revised.
  3. Rewrites LEDGER.md with the season-to-date record.

Everything here is paper. Nothing places bets, and nothing should until
the ledger itself says the edge is real.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from features import build_dataset
from parlay import decimal_odds, predict
from train import market_probs

HERE = Path(__file__).parent
CHAMPION_FILE = HERE / "champion.json"
LEDGER_FILE = HERE / "ledger.csv"
LEDGER_MD = HERE / "LEDGER.md"

COLUMNS = [
    "game_id", "season", "week", "logged_date", "away_team", "home_team",
    "pick_team", "pick_home", "model_prob", "logged_ml", "logged_market_prob",
    "closing_ml", "closing_market_prob", "clv", "result", "profit_units",
]


def side_ml(row, home: bool) -> float:
    return row["home_moneyline"] if home else row["away_moneyline"]


def load_ledger() -> pd.DataFrame:
    if LEDGER_FILE.exists():
        return pd.read_csv(LEDGER_FILE)
    return pd.DataFrame(columns=COLUMNS)


def grade_pending(ledger: pd.DataFrame, games: pd.DataFrame) -> int:
    """Settle logged picks whose games have been played."""
    played = games[games["home_win"].notna()].set_index("game_id")
    played["market_home"] = market_probs(played)

    pending = ledger["result"].eq("pending") & ledger["game_id"].isin(played.index)
    for i in ledger.index[pending]:
        game = played.loc[ledger.at[i, "game_id"]]
        pick_home = bool(ledger.at[i, "pick_home"])

        close_ml = side_ml(game, pick_home)
        if pd.notna(close_ml):
            close_prob = game["market_home"] if pick_home else 1 - game["market_home"]
            ledger.at[i, "closing_ml"] = close_ml
            ledger.at[i, "closing_market_prob"] = round(close_prob, 4)
            ledger.at[i, "clv"] = round(
                close_prob - ledger.at[i, "logged_market_prob"], 4)

        won = bool(game["home_win"]) == pick_home
        ledger.at[i, "result"] = "win" if won else "loss"
        ledger.at[i, "profit_units"] = round(
            decimal_odds(ledger.at[i, "logged_ml"]) - 1 if won else -1.0, 3)
    return int(pending.sum())


def log_new_picks(ledger: pd.DataFrame, games: pd.DataFrame,
                  config: dict) -> pd.DataFrame:
    """Log the model's pick for each priced game of the next scheduled week."""
    played = games[games["home_win"].notna()]
    future = games[games["home_win"].isna()]
    if future.empty:
        return ledger

    season = future["season"].min()
    week = future.loc[future["season"] == season, "week"].min()
    target = future[(future["season"] == season) & (future["week"] == week)]
    target = target.dropna(subset=["home_moneyline", "away_moneyline"])
    target = target[~target["game_id"].isin(ledger["game_id"])].copy()
    if target.empty:
        return ledger

    target["model_home"] = predict(config, played, target)
    target["market_home"] = market_probs(target)

    rows = []
    for g in target.itertuples():
        pick_home = g.model_home >= 0.5
        rows.append({
            "game_id": g.game_id, "season": g.season, "week": g.week,
            "logged_date": date.today().isoformat(),
            "away_team": g.away_team, "home_team": g.home_team,
            "pick_team": g.home_team if pick_home else g.away_team,
            "pick_home": pick_home,
            "model_prob": round(g.model_home if pick_home else 1 - g.model_home, 4),
            "logged_ml": g.home_moneyline if pick_home else g.away_moneyline,
            "logged_market_prob": round(
                g.market_home if pick_home else 1 - g.market_home, 4),
            "result": "pending",
        })
    print(f"Logged {len(rows)} picks for season {season} week {week}")
    return pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)


def write_summary(ledger: pd.DataFrame):
    graded = ledger[ledger["result"].isin(["win", "loss"])]
    pending = ledger[ledger["result"] == "pending"]
    lines = [
        "# Paper-Trade Ledger",
        "",
        "*Maintained automatically. 1 unit per pick at the moneyline "
        "available when logged. No real money — that's the point: this "
        "ledger is the evidence that decides whether real money ever "
        "makes sense.*",
        "",
    ]
    if not graded.empty:
        wins = (graded["result"] == "win").sum()
        profit = graded["profit_units"].sum()
        clv = graded["clv"].dropna()
        lines += [
            "## Season to date",
            "",
            f"- **Record:** {wins}-{len(graded) - wins} "
            f"({wins / len(graded):.0%})",
            f"- **Profit:** {profit:+.1f} units on {len(graded)} staked "
            f"(ROI {profit / len(graded):+.0%})",
        ]
        if not clv.empty:
            lines += [
                f"- **CLV:** {clv.mean():+.2%} average, "
                f"{(clv > 0).mean():.0%} of picks beat the close",
                "",
                "CLV (closing line value) is the tell: positive means the "
                "market moved toward our picks after we logged them. "
                "Sustained positive CLV is the only result here that would "
                "justify real stakes — a winning record without it is "
                "variance.",
            ]
        lines.append("")
    if not pending.empty:
        lines += [
            f"## Pending ({len(pending)} picks)",
            "",
            pending[["game_id", "logged_date", "pick_team", "model_prob",
                     "logged_ml"]].to_markdown(index=False),
            "",
        ]
    if graded.empty and pending.empty:
        lines += ["No picks logged yet — lines for the next week haven't "
                  "been posted.", ""]
    LEDGER_MD.write_text("\n".join(lines))


def main():
    config = json.loads(CHAMPION_FILE.read_text())["config"]
    games = build_dataset(elo_k=config["elo_k"], mov=config["mov"],
                          include_upcoming=True)

    ledger = load_ledger()
    graded = grade_pending(ledger, games)
    if graded:
        print(f"Graded {graded} picks")
    ledger = log_new_picks(ledger, games, config)

    ledger.to_csv(LEDGER_FILE, index=False, columns=COLUMNS)
    write_summary(ledger)
    print(f"Ledger: {len(ledger)} picks total "
          f"({(ledger['result'] == 'pending').sum()} pending). "
          f"Summary in {LEDGER_MD.name}")


if __name__ == "__main__":
    main()
