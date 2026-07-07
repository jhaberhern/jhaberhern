"""Export dashboard data to docs/data.json for the GitHub Pages UI.

Collects everything the static dashboard needs into one JSON file:
champion metrics, tournament history, the current week's picks (from the
paper-trade ledger), the most-likely 3-leg parlay, and the season-to-date
ledger record. The weekly Action reruns this after the tournament and
ledger update, so the page stays current without a server.
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from parlay import decimal_odds

HERE = Path(__file__).parent
DOCS = HERE.parent / "docs"


def ledger_sections(ledger: pd.DataFrame) -> tuple[dict, list, dict | None]:
    """Season-to-date summary, current week's picks, and the parlay."""
    graded = ledger[ledger["result"].isin(["win", "loss"])]
    summary = {"graded": len(graded), "pending": int((ledger["result"] == "pending").sum())}
    if len(graded):
        clv = graded["clv"].dropna()
        summary.update({
            "wins": int((graded["result"] == "win").sum()),
            "profit_units": round(float(graded["profit_units"].sum()), 2),
            "roi": round(float(graded["profit_units"].sum()) / len(graded), 4),
            "avg_clv": round(float(clv.mean()), 4) if len(clv) else None,
            "clv_beat_rate": round(float((clv > 0).mean()), 4) if len(clv) else None,
        })

    pending = ledger[ledger["result"] == "pending"]
    picks, parlay = [], None
    if len(pending):
        season = pending["season"].max()
        week = pending.loc[pending["season"] == season, "week"].max()
        wk = pending[(pending["season"] == season) & (pending["week"] == week)]
        wk = wk.sort_values("model_prob", ascending=False)
        picks = [{
            "matchup": f"{g.away_team} @ {g.home_team}",
            "pick": g.pick_team,
            "prob": round(float(g.model_prob), 4),
            "ml": int(g.logged_ml),
            "season": int(g.season),
            "week": int(g.week),
        } for g in wk.itertuples()]

        legs = wk.head(3)
        prob = float(legs["model_prob"].prod())
        payout = float(pd.Series(
            [decimal_odds(ml) for ml in legs["logged_ml"]]).prod())
        parlay = {
            "legs": list(legs["pick_team"]),
            "prob": round(prob, 4),
            "payout": round(payout, 2),
            "ev": round(prob * payout - 1, 4),
        }
    return summary, picks, parlay


def main():
    champion = json.loads((HERE / "champion.json").read_text())
    history = pd.read_csv(HERE / "history.csv")
    ledger = pd.read_csv(HERE / "ledger.csv")
    summary, picks, parlay = ledger_sections(ledger)

    data = {
        "generated": date.today().isoformat(),
        "champion": champion,
        "history": history.to_dict(orient="records"),
        "ledger": summary,
        "picks": picks,
        "parlay": parlay,
    }
    DOCS.mkdir(exist_ok=True)
    (DOCS / "data.json").write_text(json.dumps(data, indent=1) + "\n")
    print(f"Wrote {DOCS / 'data.json'} "
          f"({len(picks)} picks, {len(data['history'])} history rows)")


if __name__ == "__main__":
    main()
