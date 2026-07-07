"""Fetch and prune player-level data for the props models.

Downloads from nflverse releases (weekly player stats, snap counts,
injuries), keeps only the seasons and columns the projection models use,
and writes compressed CSVs to data-cache/. The cache is committed to the
repo: release downloads aren't reachable from every dev environment, but
raw repo files are — so the GitHub Action does the fetching and everyone
else reads the cache.

Run by .github/workflows/fetch-props-data.yml (and weekly in season).
"""
import gzip
import io
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
CACHE = HERE / "data-cache"
BASE = "https://github.com/nflverse/nflverse-data/releases/download"
FIRST_SEASON = 2016
LAST_SEASON = 2026

STAT_COLS = [
    "player_id", "player_display_name", "position", "recent_team",
    "season", "week", "season_type", "opponent_team",
    "completions", "attempts", "passing_yards", "passing_tds", "interceptions",
    "carries", "rushing_yards", "rushing_tds",
    "targets", "receptions", "receiving_yards", "receiving_tds",
    "target_share", "air_yards_share",
]


def fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=120) as r:
            return r.read()
    except Exception as e:
        print(f"  miss: {url} ({e})")
        return None


def read_csv_bytes(raw: bytes, gzipped: bool) -> pd.DataFrame:
    if gzipped:
        raw = gzip.decompress(raw)
    return pd.read_csv(io.BytesIO(raw), low_memory=False)


def load_weekly_stats() -> pd.DataFrame:
    """Weekly offense stats. nflverse has shipped these under a couple of
    naming schemes over the years — try newest first."""
    frames = []
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        for url, gz in [
            (f"{BASE}/player_stats/player_stats_{year}.csv.gz", True),
            (f"{BASE}/player_stats/stats_player_week_{year}.csv.gz", True),
            (f"{BASE}/player_stats/stats_player_week_{year}.csv", False),
        ]:
            raw = fetch(url)
            if raw:
                frames.append(read_csv_bytes(raw, gz))
                print(f"  ok: {url}")
                break
    if not frames:  # final fallback: the single combined historical file
        raw = fetch(f"{BASE}/player_stats/player_stats.csv.gz")
        if raw:
            df = read_csv_bytes(raw, True)
            frames.append(df[df["season"] >= FIRST_SEASON])
    if not frames:
        raise RuntimeError("No weekly player stats reachable")
    df = pd.concat(frames, ignore_index=True)
    # Column names drifted across schemes; normalize the ones we need
    renames = {"player_current_team": "recent_team", "team": "recent_team",
               "player_name": "player_display_name"}
    for old, new in renames.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})
    keep = [c for c in STAT_COLS if c in df.columns]
    return df[keep]


def load_per_year(release: str, keep: list[str]) -> pd.DataFrame:
    frames = []
    for year in range(FIRST_SEASON, LAST_SEASON + 1):
        raw = fetch(f"{BASE}/{release}/{release}_{year}.csv")
        if raw:
            df = read_csv_bytes(raw, False)
            frames.append(df[[c for c in keep if c in df.columns]])
    if not frames:
        raise RuntimeError(f"No {release} data reachable")
    return pd.concat(frames, ignore_index=True)


def main():
    CACHE.mkdir(exist_ok=True)

    print("Weekly player stats:")
    stats = load_weekly_stats()
    stats.to_csv(CACHE / "player_weekly.csv.gz", index=False)
    print(f"  -> {len(stats)} rows, seasons "
          f"{stats.season.min()}-{stats.season.max()}")

    print("Snap counts:")
    snaps = load_per_year("snap_counts", [
        "game_id", "season", "week", "player", "pfr_player_id", "position",
        "team", "opponent", "offense_snaps", "offense_pct"])
    snaps.to_csv(CACHE / "snap_counts.csv.gz", index=False)
    print(f"  -> {len(snaps)} rows")

    print("Injuries:")
    inj = load_per_year("injuries", [
        "season", "week", "team", "gsis_id", "full_name", "position",
        "report_status", "practice_status", "date_modified"])
    inj.to_csv(CACHE / "injuries.csv.gz", index=False)
    print(f"  -> {len(inj)} rows")


if __name__ == "__main__":
    main()
