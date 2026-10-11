"""Synthetic retail demand: many series with trend, weekly/yearly seasonality, promos, holidays."""
from __future__ import annotations
import datetime as dt
import numpy as np
import polars as pl

HOLIDAYS = [(1, 1), (2, 14), (5, 1), (8, 15), (10, 31), (11, 25), (12, 24), (12, 25)]


def make_demand(n_series: int = 40, start: str = "2022-01-01", days: int = 1096, seed: int = 1) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    dates = np.arange(np.datetime64(start), np.datetime64(start) + days)
    t = np.arange(days)
    dow = ((dates.astype("datetime64[D]").astype(int) + 3) % 7)  # 0=Mon
    doy = (dates - dates.astype("datetime64[Y]")).astype(int)
    md = np.array([(d.month, d.day) for d in dates.astype(dt.date)])
    is_hol = np.isin(md[:, 0] * 100 + md[:, 1], [m * 100 + d for m, d in HOLIDAYS]).astype(int)

    frames = []
    for s in range(n_series):
        level = rng.lognormal(3.0, 0.7)
        trend = 1 + rng.normal(0.0004, 0.0003) * t
        weekly = 1 + rng.uniform(0.1, 0.5) * np.sin(2 * np.pi * (dow - rng.integers(0, 7)) / 7)
        yearly = 1 + rng.uniform(0.05, 0.35) * np.sin(2 * np.pi * (doy - rng.integers(0, 365)) / 365.25)
        promo = (rng.random(days) < 0.08).astype(int)
        mu = level * trend * weekly * yearly * (1 + 0.45 * promo) * (1 + 0.6 * is_hol)
        sales = rng.negative_binomial(n=12, p=12 / (12 + mu))
        frames.append(pl.DataFrame({
            "series_id": f"S{s:03d}", "date": dates.astype("datetime64[us]"),
            "sales": sales.astype(float), "promo": promo, "is_holiday": is_hol}))
    return pl.concat(frames)
