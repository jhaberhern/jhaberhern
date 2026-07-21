"""Archive Kalshi market prices — the read-only Gate-1 step for trading
event contracts.

Kalshi is a CFTC-regulated exchange, not a sportsbook: prices are a
peer-to-peer order book quoted 1-99c on contracts that settle at $1, so
a price of 62c *is* a 62% implied probability — no vig to strip. Unlike
a sportsbook it has an official API and permits algorithmic trading, so
this is the one venue where the endgame (a model that funds itself) can
be closed legally and in-terms. But first we observe: this collector
snapshots live prices so any future model can be graded against the
market before a cent is at risk.

Read-only market data needs no credentials. We poll public endpoints,
keep the fields a model cares about (bid/ask/last/volume/open interest),
tag each row with a UTC snapshot time, and append to the archive. By
watching openers vs. closes we also build the same CLV dataset that the
sports side relies on.

Which markets: by default we cast wide across categories (weather,
economics, sports, etc.) and record where the liquidity and movement
actually are — that's how we decide what's worth modeling. Narrow later
with --series once a beachhead is chosen.

Run by .github/workflows/collect-kalshi.yml and the self-hosted server.
"""
import argparse
import gzip  # noqa: F401  (kept for parity with other collectors' IO)
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "data-cache" / "kalshi_prices.csv.gz"
# Kalshi unified production API (v2). Override with KALSHI_API_BASE if the
# host changes.
BASE = os.environ.get(
    "KALSHI_API_BASE", "https://api.elections.kalshi.com/trade-api/v2")

COLUMNS = [
    "snapshot_utc", "ticker", "event_ticker", "series_ticker", "title",
    "category", "status", "close_time",
    "yes_bid", "yes_ask", "last_price", "volume", "open_interest",
    "liquidity",
]


def get(path: str, **params) -> dict:
    url = f"{BASE}{path}"
    if params:
        url += "?" + urllib.parse.urlencode({k: v for k, v in params.items()
                                             if v is not None})
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read())


def fetch_markets(status: str, series: str | None, max_pages: int) -> list[dict]:
    """Paginate the public /markets endpoint. Kalshi returns a cursor;
    an empty cursor means we're done."""
    out, cursor, pages = [], None, 0
    while pages < max_pages:
        page = get("/markets", limit=200, status=status,
                   series_ticker=series, cursor=cursor)
        out += page.get("markets", [])
        cursor = page.get("cursor")
        pages += 1
        if not cursor:
            break
        time.sleep(0.3)  # be polite to the public endpoint
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--status", default="open",
                    help="market status filter (default: open)")
    ap.add_argument("--series", default=None,
                    help="restrict to one series ticker (e.g. weather)")
    ap.add_argument("--max-pages", type=int, default=25,
                    help="pagination cap (200 markets/page)")
    args = ap.parse_args()

    snapshot = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        markets = fetch_markets(args.status, args.series, args.max_pages)
    except Exception as e:
        # Unreachable (sandbox/policy) or transient: never fail the pipeline
        print(f"Kalshi API unreachable ({e}); nothing collected this run. "
              "This is expected in environments that can't reach Kalshi.")
        return

    rows = [{
        "snapshot_utc": snapshot,
        "ticker": m.get("ticker"),
        "event_ticker": m.get("event_ticker"),
        "series_ticker": m.get("series_ticker"),
        "title": m.get("title"),
        "category": m.get("category"),
        "status": m.get("status"),
        "close_time": m.get("close_time"),
        "yes_bid": m.get("yes_bid"),
        "yes_ask": m.get("yes_ask"),
        "last_price": m.get("last_price"),
        "volume": m.get("volume"),
        "open_interest": m.get("open_interest"),
        "liquidity": m.get("liquidity"),
    } for m in markets]
    print(f"{len(rows)} Kalshi markets snapshotted "
          f"(status={args.status}, series={args.series or 'all'})")
    if not rows:
        return

    new = pd.DataFrame(rows, columns=COLUMNS)
    if OUT.exists():
        new = pd.concat([pd.read_csv(OUT), new], ignore_index=True)
    # One row per market per snapshot; keep the newest if a run overlaps
    new = new.drop_duplicates(["snapshot_utc", "ticker"], keep="last")
    OUT.parent.mkdir(exist_ok=True)
    new.to_csv(OUT, index=False)

    cats = (pd.Series([r["category"] for r in rows]).value_counts()
            .head(6).to_dict())
    print(f"Archive now {len(new)} rows -> {OUT.name}")
    print(f"Categories this snapshot: {cats}")


if __name__ == "__main__":
    main()
