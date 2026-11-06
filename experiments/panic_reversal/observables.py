"""Causal daily / incremental-trade features, without assigning panic truth labels.

Daily schema: date,asset,close,adj_close,volume,amount.
Trade schema: timestamp,asset,price,volume,aggressor (-1/0/1 or blank).
Tick timestamps must include an offset. Volume is incremental, never cumulative.
"""
import argparse
from collections import defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import math
from pathlib import Path

import numpy as np

from .run import write_csv


SHANGHAI = timezone(timedelta(hours=8))


def number(row, key, positive=False):
    value = float(row[key])
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f"Invalid {key}")
    return value


def daily_features(rows):
    """Use adjacent market dates only; missing returns stay missing, never zero.

    Breadth uses securities with both dates available. coverage exposes changes
    in that denominator; a point-in-time universe is the caller's responsibility.
    """
    days = defaultdict(dict)
    for row in rows:
        day = date.fromisoformat(row["date"])
        code = row["asset"]
        if not code or code in days[day]:
            raise ValueError("Missing code or duplicate asset/date")
        days[day][code] = {key: number(row, key, key in ("close", "adj_close")) for key in ("close", "adj_close", "volume", "amount")}
    result, previous, amounts, returns = [], {}, [], []
    for day, records in sorted(days.items()):
        ret = [r["adj_close"] / previous[s]["adj_close"] - 1 for s, r in records.items() if s in previous]
        turnover = sum(r["amount"] for r in records.values())
        rmarket = float(np.mean(ret)) if ret else None
        prior = np.log1p(amounts[-20:])
        volume_z = float((np.log1p(turnover) - prior.mean()) / prior.std(ddof=1)) if len(prior) == 20 and prior.std(ddof=1) > 1e-10 else None
        recent_returns = (returns + [rmarket])[-20:]
        vol = float(np.std(recent_returns, ddof=1)) if len(recent_returns) == 20 and all(v is not None for v in recent_returns) else None
        # Daily data cannot identify initiator side, account identity or diffusion.
        result.append({"date": day.isoformat(), "n_assets": len(records), "n_return_assets": len(ret),
                       "coverage": len(ret) / len(records), "market_return_equal_weight": rmarket,
                       "breadth": float(np.mean(np.asarray(ret) > 0)) if ret else None,
                       "return_dispersion": float(np.std(ret)) if ret else None,
                       "market_vol20": vol, "turnover_amount": turnover, "volume_z20": volume_z,
                       "amihud_market_proxy": abs(rmarket) / turnover if rmarket is not None and turnover > 0 else None})
        previous, amounts, returns = records, amounts + [turnover], returns + [rmarket]
    return result


def tick_features(rows, bucket_minutes=5):
    """Finished buckets only; no future daily totals are copied into intraday rows.

    aggressor is supplied by the data provider. Blank/0 means unknown. Coverage
    is known-side notional / all trade notional; unknown trades are not guessed.
    End-of-bucket price changes are contemporaneous impact proxies, not causal
    estimates of the price impact of any individual order.
    """
    if type(bucket_minutes) is not int or not 1 <= bucket_minutes <= 60:
        raise ValueError("bucket_minutes must be in [1,60]")
    buckets = defaultdict(list)
    for seq, row in enumerate(rows):
        timestamp = datetime.fromisoformat(row["timestamp"])
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("Tick timestamp needs an explicit timezone offset")
        timestamp = timestamp.astimezone(SHANGHAI)
        minute = timestamp.hour * 60 + timestamp.minute
        start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=(minute // bucket_minutes) * bucket_minutes)
        price, volume = number(row, "price", True), number(row, "volume")
        raw = row.get("aggressor", "")
        sign = 0 if raw is None or raw == "" else int(raw)
        if str(raw) not in ("", "None", "-1", "0", "1") or sign not in (-1, 0, 1):
            raise ValueError("aggressor must be -1, 0, 1 or blank")
        if not row["asset"]:
            raise ValueError("Missing asset code")
        buckets[start].append((timestamp, seq, row["asset"], price, volume, sign))
    result, previous_prices, sell_history = [], {}, defaultdict(list)
    for start, trades in sorted(buckets.items()):
        by_asset = defaultdict(lambda: {"amount": 0.0, "sell": 0.0, "buy": 0.0})
        last = {}
        for _, _, code, price, volume, sign in sorted(trades):
            amount = price * volume
            by_asset[code]["amount"] += amount
            if sign == -1:
                by_asset[code]["sell"] += amount
            if sign == 1:
                by_asset[code]["buy"] += amount
            last[code] = price
        ret = [price / previous_prices[s] - 1 for s, price in last.items() if s in previous_prices]
        buy, sell, total = (sum(r[k] for r in by_asset.values()) for k in ("buy", "sell", "amount"))
        sell_history[start.date()].append(sell)
        sell_array = np.asarray(sell_history[start.date()])
        result.append({"bucket_start": start.isoformat(), "bucket_end": (start + timedelta(minutes=bucket_minutes)).isoformat(),
                       "n_assets": len(last), "n_price_change_assets": len(ret), "turnover_amount": total,
                       "aggressor_coverage": (buy + sell) / total if total > 0 else None,
                       "aggressive_sell_share": sell / (buy + sell) if buy + sell > 0 else None,
                       "signed_sell_imbalance": (sell - buy) / (sell + buy) if sell + buy > 0 else None,
                       "sell_concentration_assets": sum((r["sell"] / sell)**2 for r in by_asset.values()) if sell > 0 else None,
                       "sell_time_hhi_day_to_date": float(np.square(sell_array / sell_array.sum()).sum()) if sell_array.sum() > 0 else None,
                       "mean_price_change": float(np.mean(ret)) if ret else None,
                       "breadth": float(np.mean(np.asarray(ret) > 0)) if ret else None,
                       "return_dispersion": float(np.std(ret)) if ret else None,
                       "amihud_bucket_proxy": abs(float(np.mean(ret))) / total if ret and total > 0 else None})
        # Compare with last observed prices: missing symbols are disclosed in coverage.
        previous_prices.update(last)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frequency", choices=("daily", "tick"))
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bucket-minutes", default=5, type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Choose a new output path")
    with args.input.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    features = daily_features(rows) if args.frequency == "daily" else tick_features(rows, args.bucket_minutes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.output, features)
    print(f"wrote {len(features)} completed {args.frequency} observations; no panic-type labels inferred")


if __name__ == "__main__":
    main()
