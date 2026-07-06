"""Download the nflverse games dataset (every NFL game since 1999).

Includes final scores, rest days, and the Vegas lines for each game,
which lets us compare our model against the betting market.
"""
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
DEST = Path(__file__).parent / "data" / "games.csv"


def main():
    DEST.parent.mkdir(exist_ok=True)
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DEST)
    size_kb = DEST.stat().st_size // 1024
    print(f"Saved {DEST} ({size_kb} KB)")


if __name__ == "__main__":
    main()
