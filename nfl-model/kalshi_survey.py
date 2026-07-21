"""Survey the Kalshi price archive to find where an edge might live.

Before modeling anything, we want to know which categories are liquid
enough to trade and volatile enough to be worth predicting. This reads
the collected snapshots and reports, per category:

    markets      -- distinct contracts seen
    volume       -- total contracts traded (liquidity proxy)
    avg_spread   -- mean yes_ask - yes_bid in cents (tighter = more
                    efficient / cheaper to trade; wide = illiquid but
                    possibly mispriced)
    price_move   -- mean absolute change in last_price across snapshots
                    (how much prices actually move = room for a forecast
                    to matter)

High volume + meaningful price movement + a spread we can pay = a
candidate beachhead. Weather and economic-data series are the usual
winners for a public-data modeler; this tells us for sure.
"""
from pathlib import Path

import pandas as pd

ARCHIVE = Path(__file__).parent / "data-cache" / "kalshi_prices.csv.gz"


def main():
    if not ARCHIVE.exists():
        print("No Kalshi archive yet. Run kalshi_collector.py (or the "
              "collect-kalshi workflow) a few times first.")
        return
    df = pd.read_csv(ARCHIVE)
    df["spread"] = df["yes_ask"] - df["yes_bid"]

    # price movement per market across snapshots
    df = df.sort_values("snapshot_utc")
    moves = (df.groupby("ticker")["last_price"].apply(
        lambda s: s.diff().abs().mean()).rename("price_move"))
    df = df.merge(moves, on="ticker", how="left")

    by_cat = df.groupby(df["category"].fillna("(uncategorized)")).agg(
        markets=("ticker", "nunique"),
        total_volume=("volume", "max"),
        avg_spread_cents=("spread", "mean"),
        avg_price_move=("price_move", "mean"),
        snapshots=("snapshot_utc", "nunique"),
    ).sort_values("total_volume", ascending=False)

    with pd.option_context("display.float_format", "{:.2f}".format,
                           "display.max_rows", 40):
        print(by_cat.to_string())
    print("\nBeachhead = high volume + real price_move + a spread you can "
          "pay. Narrow the collector to that series next.")


if __name__ == "__main__":
    main()
