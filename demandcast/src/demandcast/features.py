"""
Leak-free feature engineering with Polars window expressions.

Every lag/rolling feature is shifted by at least HORIZON days, so a *direct* multi-step
forecast can be made for any day inside the horizon without recursive prediction.

"""
from __future__ import annotations
import polars as pl

HORIZON = 28
LAGS = [28, 35, 42, 49, 56, 364]


def build_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["series_id", "date"])
    s = pl.col("sales")
    exprs = [s.shift(l).over("series_id").alias(f"lag_{l}") for l in LAGS]
    shifted = s.shift(HORIZON).over("series_id")
    exprs += [
        shifted.rolling_mean(7).over("series_id").alias("roll_mean_7"),
        shifted.rolling_mean(28).over("series_id").alias("roll_mean_28"),
        shifted.rolling_std(28).over("series_id").alias("roll_std_28"),
        shifted.rolling_mean(91).over("series_id").alias("roll_mean_91"),
    ]
    out = df.with_columns(exprs).with_columns(
        pl.col("date").dt.weekday().alias("dow"),
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.ordinal_day().alias("doy"),
        pl.col("date").dt.week().alias("week"),
        pl.col("series_id").str.slice(1).cast(pl.Int32).alias("series_code"),
    ).with_columns(
        (pl.col("lag_28") + pl.col("lag_35") + pl.col("lag_42") + pl.col("lag_49")).truediv(4).alias("same_dow_mean"),
        (pl.col("roll_mean_7") / (pl.col("roll_mean_91") + 1)).alias("recent_ratio"),
    )
    return out.drop_nulls(["lag_56", "roll_mean_91"])


FEATURES = (
    [f"lag_{l}" for l in LAGS]
    + ["roll_mean_7", "roll_mean_28", "roll_std_28", "roll_mean_91", "same_dow_mean", "recent_ratio",
       "dow", "month", "doy", "week", "promo", "is_holiday", "series_code"]
)
