"""Feature builder for player prop projections.

Philosophy: model OPPORTUNITY, not outcomes. A receiver's yards bounce
around; his target share doesn't. Every feature is trailing (shifted one
game) so nothing leaks from the week being predicted — same golden rule
as the game model.

Feature groups per player-week:
  usage      -- trailing 4- and 8-game averages of the volume stats
                (targets, carries, attempts) and efficiency outcomes
  role       -- trailing target share; games played recency
  context    -- the Vegas game environment: the team's implied points
                from the spread and total (books do the macro work,
                we borrow it), home/away
  matchup    -- what the opponent's defense has allowed to this
                position, season-to-date trailing average

Markets supported: receptions, receiving_yards, rushing_yards,
passing_yards.
"""
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "data-cache"
GAMES = HERE / "data" / "games.csv"

MARKETS = {
    "receptions":      {"positions": ["WR", "TE", "RB"], "volume": "targets"},
    "receiving_yards": {"positions": ["WR", "TE", "RB"], "volume": "targets"},
    "rushing_yards":   {"positions": ["RB", "QB"],       "volume": "carries"},
    "passing_yards":   {"positions": ["QB"],             "volume": "attempts"},
}
TRAIL_STATS = ["targets", "receptions", "receiving_yards", "carries",
               "rushing_yards", "attempts", "passing_yards", "target_share"]


def load_player_weeks() -> pd.DataFrame:
    path = CACHE / "player_weekly.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing - run props_fetch.py (or the fetch-props-data "
            "workflow) to populate the data cache")
    df = pd.read_csv(path)
    df = df[df["season_type"].isin(["REG", "POST"])] if "season_type" in df else df
    return df.sort_values(["season", "week"]).reset_index(drop=True)


def game_context() -> pd.DataFrame:
    """Team-week Vegas context from the games file: implied points and
    home flag. spread_line is the expected home margin, so
    home implied = (total + spread) / 2."""
    g = pd.read_csv(GAMES)
    g = g.dropna(subset=["total_line", "spread_line"])
    home = pd.DataFrame({
        "season": g["season"], "week": g["week"], "team": g["home_team"],
        "implied_points": (g["total_line"] + g["spread_line"]) / 2,
        "is_home": 1.0,
    })
    away = pd.DataFrame({
        "season": g["season"], "week": g["week"], "team": g["away_team"],
        "implied_points": (g["total_line"] - g["spread_line"]) / 2,
        "is_home": 0.0,
    })
    return pd.concat([home, away], ignore_index=True)


def add_trailing(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing means over the previous 4 and 8 appearances (shifted so
    the current week never sees itself)."""
    df = df.sort_values(["player_id", "season", "week"])
    grp = df.groupby("player_id")
    for stat in TRAIL_STATS:
        if stat not in df:
            continue
        shifted = grp[stat].shift(1)
        for n in (4, 8):
            df[f"tr{n}_{stat}"] = (
                shifted.groupby(df["player_id"])
                .rolling(n, min_periods=1).mean()
                .reset_index(level=0, drop=True))
    df["games_prior"] = grp.cumcount()
    return df


def add_defense_allowance(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Trailing average of `market` the opponent has allowed per game to
    this position group, season to date (shifted one week)."""
    allowed = (df.groupby(["season", "week", "opponent_team", "position"])
               [market].sum().reset_index()
               .rename(columns={market: "allowed", "opponent_team": "def_team"}))
    allowed = allowed.sort_values(["season", "week"])
    grp = allowed.groupby(["season", "def_team", "position"])
    allowed["def_allowed"] = (grp["allowed"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()))
    return df.merge(
        allowed[["season", "week", "def_team", "position", "def_allowed"]],
        left_on=["season", "week", "opponent_team", "position"],
        right_on=["season", "week", "def_team", "position"], how="left"
    ).drop(columns="def_team")


def build_market_dataset(market: str) -> tuple[pd.DataFrame, list[str]]:
    """Feature matrix + feature-column names for one prop market."""
    spec = MARKETS[market]
    df = load_player_weeks()
    df = df[df["position"].isin(spec["positions"])].copy()

    # Only player-weeks with a real role: saves the model from special-
    # teamers and inactive weeks (props aren't posted for those anyway)
    df = add_trailing(df)
    df = df[df[f"tr4_{spec['volume']}"].fillna(0) > 1.0]
    df = df[df["games_prior"] >= 2]

    df = add_defense_allowance(df, market)
    ctx = game_context()
    df = df.merge(ctx, left_on=["season", "week", "recent_team"],
                  right_on=["season", "week", "team"], how="left")

    features = ([c for c in df.columns if c.startswith(("tr4_", "tr8_"))]
                + ["games_prior", "def_allowed", "implied_points", "is_home"])
    df = df.dropna(subset=[market])
    df[features] = df[features].fillna(-1.0)  # GBDT-friendly missing marker
    return df, features
