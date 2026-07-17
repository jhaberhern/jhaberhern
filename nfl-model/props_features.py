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
               "rushing_yards", "attempts", "passing_yards", "target_share",
               "offense_pct"]

# Which side of the opponent's defense each market attacks
MARKET_DEF_EPA = {"receptions": "def_pass_epa", "receiving_yards": "def_pass_epa",
                  "passing_yards": "def_pass_epa", "rushing_yards": "def_rush_epa"}


def norm_name(s: str) -> str:
    import re
    s = re.sub(r"[^a-z ]", "", str(s).lower())
    return " ".join(s.split()[:2])


def add_snap_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Attach each player-week's offensive snap share (by name+team+week —
    the snap file uses PFR ids, the stats file GSIS ids). Snap share is
    the purest role signal there is; its trailing values are computed by
    add_trailing like every other stat."""
    path = CACHE / "snap_counts.csv.gz"
    if not path.exists():
        df["offense_pct"] = float("nan")
        return df
    snaps = pd.read_csv(path)
    snaps["norm_name"] = snaps["player"].map(norm_name)
    snaps = (snaps.groupby(["season", "week", "team", "norm_name"],
                           as_index=False)["offense_pct"].max())
    df = df.copy()
    df["norm_name"] = df["player_display_name"].map(norm_name)
    df = df.merge(snaps, left_on=["season", "week", "recent_team", "norm_name"],
                  right_on=["season", "week", "team", "norm_name"],
                  how="left").drop(columns=["team"])
    return df


def add_injuries(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-game injury-report features — the timing edge. A player's own
    Questionable/Doubtful tag, and how many same-position teammates are
    ruled Out (their volume has to go somewhere)."""
    path = CACHE / "injuries.csv.gz"
    df = df.copy()
    if not path.exists():
        df["own_questionable"] = 0.0
        df["teammates_out_pos"] = 0.0
        return df
    inj = pd.read_csv(path)

    own = inj[inj["report_status"].isin(["Questionable", "Doubtful"])]
    own = own[["season", "week", "gsis_id"]].drop_duplicates()
    own["own_questionable"] = 1.0
    df = df.merge(own, left_on=["season", "week", "player_id"],
                  right_on=["season", "week", "gsis_id"], how="left"
                  ).drop(columns=["gsis_id"])

    out = (inj[inj["report_status"] == "Out"]
           .groupby(["season", "week", "team", "position"])
           .size().rename("teammates_out_pos").reset_index())
    df = df.merge(out, left_on=["season", "week", "recent_team", "position"],
                  right_on=["season", "week", "team", "position"], how="left"
                  ).drop(columns=["team"])
    df["own_questionable"] = df["own_questionable"].fillna(0.0)
    df["teammates_out_pos"] = df["teammates_out_pos"].fillna(0.0)
    return df


def add_vacated_volume(df: pd.DataFrame, all_stats: pd.DataFrame) -> pd.DataFrame:
    """How much target volume a team's ruled-Out players take with them
    this week. A WR2 gains far more when the WR1's nine targets vanish
    than when a depth receiver sits — the count of Outs can't see that;
    the sum of their recent usage can. Strictly pre-game: uses each Out
    player's average targets over games already played."""
    path = CACHE / "injuries.csv.gz"
    df = df.copy()
    if not path.exists():
        df["vacated_targets"] = 0.0
        return df
    inj = pd.read_csv(path)
    outs = inj[inj["report_status"] == "Out"][
        ["season", "week", "team", "gsis_id"]].drop_duplicates()

    # Each player's season-to-date average targets AFTER each game played,
    # then carried forward to any later week via a backward asof-join
    hist = all_stats[["player_id", "season", "week", "targets"]].copy()
    hist["order"] = hist["season"] * 100 + hist["week"]
    hist = hist.sort_values("order")
    hist["avg_targets"] = (hist.groupby("player_id")["targets"]
                           .transform(lambda s: s.expanding(min_periods=1).mean()))
    outs = outs.copy()
    outs["order"] = outs["season"] * 100 + outs["week"]
    outs = pd.merge_asof(
        outs.sort_values("order"), hist.sort_values("order"),
        on="order", left_by="gsis_id", right_by="player_id",
        allow_exact_matches=False, direction="backward",
        suffixes=("", "_h"))

    vac = (outs.groupby(["season", "week", "team"])["avg_targets"]
           .sum().rename("vacated_targets").reset_index())
    df = df.merge(vac, left_on=["season", "week", "recent_team"],
                  right_on=["season", "week", "team"], how="left"
                  ).drop(columns=["team"])
    df["vacated_targets"] = df["vacated_targets"].fillna(0.0)
    return df


def add_opp_def_epa(df: pd.DataFrame, market: str) -> pd.DataFrame:
    """Opponent's season-to-date defensive EPA/play on the relevant side
    (pass or rush), shifted one week so it's strictly pre-game."""
    path = CACHE / "team_week_epa.csv.gz"
    col = MARKET_DEF_EPA[market]
    df = df.copy()
    if not path.exists():
        df["opp_def_epa"] = float("nan")
        return df
    epa = pd.read_csv(path).sort_values(["season", "week"])
    epa["opp_def_epa"] = (epa.groupby(["season", "team"])[col]
                          .transform(lambda s: s.shift(1)
                                     .expanding(min_periods=1).mean()))
    return df.merge(
        epa[["season", "week", "team", "opp_def_epa"]],
        left_on=["season", "week", "opponent_team"],
        right_on=["season", "week", "team"], how="left").drop(columns=["team"])


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
    df = add_snap_pct(df)
    df = add_trailing(df)
    df = df[df[f"tr4_{spec['volume']}"].fillna(0) > 1.0]
    df = df[df["games_prior"] >= 2]
    df["snap_trend"] = df["tr4_offense_pct"] - df["tr8_offense_pct"]

    df = add_injuries(df)
    df = add_vacated_volume(df, load_player_weeks())
    df = add_opp_def_epa(df, market)
    df = add_defense_allowance(df, market)
    ctx = game_context()
    df = df.merge(ctx, left_on=["season", "week", "recent_team"],
                  right_on=["season", "week", "team"], how="left")

    features = ([c for c in df.columns if c.startswith(("tr4_", "tr8_"))]
                + ["games_prior", "def_allowed", "implied_points", "is_home",
                   "snap_trend", "own_questionable", "teammates_out_pos",
                   "vacated_targets", "opp_def_epa"])
    df = df.dropna(subset=[market])
    df[features] = df[features].fillna(-1.0)  # GBDT-friendly missing marker
    return df, features
