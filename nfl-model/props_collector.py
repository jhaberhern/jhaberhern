"""Archive player prop lines from The Odds API.

Historical prop lines are not publicly available anywhere — so we build
our own archive. Run on a schedule, this snapshots current prop lines
(with timestamps) into data-cache/prop_lines.csv.gz. Early-week and
near-kickoff snapshots together give us openers AND closers: the raw
material for real CLV measurement on props.

Setup (one-time):
  1. Free key from https://the-odds-api.com
  2. Repo Settings -> Secrets and variables -> Actions -> new secret
     named ODDS_API_KEY
  3. The collector workflow picks it up from the environment.

Frugality: the free tier is credit-limited and prop markets bill per
market x region. We poll one region, a short bookmaker list, and only
the markets we model. Roughly: 1 events call + 1 odds call per game
week, ~2 snapshots a week in season — comfortably inside the free tier.
"""
import gzip
import io
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "data-cache" / "prop_lines.csv.gz"
OUT_GAMES = HERE / "data-cache" / "game_lines.csv.gz"
BASE = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl"

MARKETS = ["player_receptions", "player_reception_yds",
           "player_rush_yds", "player_pass_yds"]
BOOKMAKERS = ["hardrockbet", "draftkings", "fanduel"]

COLUMNS = ["snapshot_utc", "event_id", "commence_time", "home_team",
           "away_team", "bookmaker", "market", "player", "line",
           "over_price", "under_price"]


def get(path: str, **params) -> tuple[list | dict, dict]:
    params["apiKey"] = os.environ["ODDS_API_KEY"]
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:
        remaining = r.headers.get("x-requests-remaining")
        return json.loads(r.read()), {"remaining": remaining}


def flatten(event: dict, snapshot: str) -> list[dict]:
    rows = []
    for bk in event.get("bookmakers", []):
        for mkt in bk.get("markets", []):
            # group over/under outcomes by (player, line)
            legs: dict[tuple, dict] = {}
            for o in mkt.get("outcomes", []):
                key = (o.get("description"), o.get("point"))
                legs.setdefault(key, {})[o.get("name")] = o.get("price")
            for (player, line), prices in legs.items():
                rows.append({
                    "snapshot_utc": snapshot,
                    "event_id": event["id"],
                    "commence_time": event.get("commence_time"),
                    "home_team": event.get("home_team"),
                    "away_team": event.get("away_team"),
                    "bookmaker": bk["key"],
                    "market": mkt["key"],
                    "player": player,
                    "line": line,
                    "over_price": prices.get("Over"),
                    "under_price": prices.get("Under"),
                })
    return rows


def collect_game_lines(snapshot: str):
    """Snapshot game-level odds (moneyline/spread/total) — one bulk call,
    ~3 credits. nflverse only preserves closing lines, so archiving our
    own openers gives the game model a real open-to-close CLV dataset."""
    events, quota = get("/odds", regions="us", markets="h2h,spreads,totals",
                        bookmakers=",".join(BOOKMAKERS), oddsFormat="american")
    rows = []
    for ev in events:
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                for o in mkt.get("outcomes", []):
                    rows.append({
                        "snapshot_utc": snapshot, "event_id": ev["id"],
                        "commence_time": ev.get("commence_time"),
                        "home_team": ev.get("home_team"),
                        "away_team": ev.get("away_team"),
                        "bookmaker": bk["key"], "market": mkt["key"],
                        "name": o.get("name"), "point": o.get("point"),
                        "price": o.get("price"),
                    })
    print(f"{len(rows)} game lines collected "
          f"(credits left: {quota['remaining']})")
    if not rows:
        return
    new = pd.DataFrame(rows)
    if OUT_GAMES.exists():
        new = pd.concat([pd.read_csv(OUT_GAMES), new],
                        ignore_index=True).drop_duplicates()
    new.to_csv(OUT_GAMES, index=False)
    print(f"Game-line archive now {len(new)} rows -> {OUT_GAMES.name}")


def main():
    if not os.environ.get("ODDS_API_KEY"):
        print("ODDS_API_KEY not set - nothing to collect. See module "
              "docstring for setup; this is expected until the key exists.")
        return

    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    collect_game_lines(snapshot)
    events, quota = get("/events")
    print(f"{len(events)} upcoming events (credits left: {quota['remaining']})")

    rows = []
    for ev in events:
        odds, quota = get(f"/events/{ev['id']}/odds", regions="us",
                          markets=",".join(MARKETS),
                          bookmakers=",".join(BOOKMAKERS),
                          oddsFormat="american")
        rows += flatten(odds, snapshot)
    print(f"{len(rows)} prop lines collected "
          f"(credits left: {quota['remaining']})")
    if not rows:
        return

    new = pd.DataFrame(rows, columns=COLUMNS)
    if OUT.exists():
        old = pd.read_csv(OUT)
        new = pd.concat([old, new], ignore_index=True).drop_duplicates()
    new.to_csv(OUT, index=False)
    print(f"Archive now {len(new)} rows -> {OUT.name}")


if __name__ == "__main__":
    main()
