"""Build pre-game features for every NFL game.

The golden rule here is NO DATA LEAKAGE: every feature must be something
you could have known *before kickoff*. That's why Elo ratings are computed
by walking through games in order — each game is rated using only the
games played before it.

Features produced:
    elo_diff   -- home team Elo minus away team Elo, before the game
    rest_diff  -- home team days of rest minus away team days of rest
    div_game   -- 1 if it's a divisional matchup (they tend to be closer)

Target:
    home_win   -- 1 if the home team won (ties are dropped; they're rare)
"""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).parent / "data" / "games.csv"

# Standard NFL Elo parameters (in the spirit of FiveThirtyEight's model)
ELO_START = 1500.0
ELO_K = 20.0            # how fast ratings react to results
HOME_FIELD_ELO = 48.0   # home advantage, in Elo points, used inside updates
SEASON_REGRESSION = 1 / 3  # each offseason, pull ratings back toward the mean


def add_elo(games: pd.DataFrame, k: float = ELO_K,
            hfa: float = HOME_FIELD_ELO,
            regression: float = SEASON_REGRESSION) -> pd.DataFrame:
    """Walk through games chronologically, tracking an Elo rating per team.

    Stores each team's rating *before* the game — that's the pre-game
    knowledge a bettor would have had.
    """
    ratings: dict[str, float] = {}
    last_season: dict[str, int] = {}
    home_elo, away_elo = [], []

    for game in games.itertuples():
        for team in (game.home_team, game.away_team):
            ratings.setdefault(team, ELO_START)
            # New season: regress toward the mean (rosters change, teams churn)
            if last_season.get(team, game.season) != game.season:
                ratings[team] += regression * (ELO_START - ratings[team])
            last_season[team] = game.season

        h, a = ratings[game.home_team], ratings[game.away_team]
        home_elo.append(h)
        away_elo.append(a)

        # Update ratings from the result (result = home score - away score)
        expected_home = 1 / (1 + 10 ** (-(h - a + hfa) / 400))
        actual_home = 0.5 if game.result == 0 else float(game.result > 0)
        shift = k * (actual_home - expected_home)
        ratings[game.home_team] += shift
        ratings[game.away_team] -= shift

    games = games.copy()
    games["home_elo"] = home_elo
    games["away_elo"] = away_elo
    games["elo_diff"] = games["home_elo"] - games["away_elo"]
    return games


def add_rolling_margin(games: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Each team's average point margin over its last `n` games, pre-game.

    Catches recent form that win/loss Elo smooths over (a team winning
    by 20 and a team winning by 1 look the same to Elo). Teams with no
    history yet get 0 (neutral).
    """
    from collections import defaultdict, deque
    recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=n))
    home_m, away_m = [], []

    for game in games.itertuples():
        h, a = recent[game.home_team], recent[game.away_team]
        home_m.append(sum(h) / len(h) if h else 0.0)
        away_m.append(sum(a) / len(a) if a else 0.0)
        h.append(game.result)       # home margin
        a.append(-game.result)      # away margin

    games = games.copy()
    games["margin_diff"] = pd.Series(home_m, index=games.index) - pd.Series(
        away_m, index=games.index)
    return games


def build_dataset(elo_k: float = ELO_K, elo_hfa: float = HOME_FIELD_ELO,
                  regression: float = SEASON_REGRESSION) -> pd.DataFrame:
    games = pd.read_csv(DATA)
    games = games[games["result"].notna()]  # played games only
    games = games.sort_values(["season", "week", "gameday"]).reset_index(drop=True)

    games = add_elo(games, k=elo_k, hfa=elo_hfa, regression=regression)
    games = add_rolling_margin(games)

    games["rest_diff"] = games["home_rest"] - games["away_rest"]
    games["home_win"] = (games["result"] > 0).astype(int)

    games = games[games["result"] != 0]  # drop ties
    keep = [
        "game_id", "season", "week", "home_team", "away_team",
        "elo_diff", "rest_diff", "div_game", "margin_diff",
        "home_moneyline", "away_moneyline",
        "home_win",
    ]
    return games[keep].dropna(subset=["elo_diff", "rest_diff"])


if __name__ == "__main__":
    df = build_dataset()
    print(df.tail(10).to_string(index=False))
    print(f"\n{len(df)} games, seasons {df.season.min()}-{df.season.max()}")
