"""Rolling-origin backtest. Per fold: train < calibration window < test window (28d each)."""
from __future__ import annotations
import numpy as np
import polars as pl
from .features import HORIZON
from .models import QuantileForecaster, seasonal_naive


def wape(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.abs(y - p).sum() / max(y.sum(), 1e-9))


def split_fold(df: pl.DataFrame, test_start, horizon: int = HORIZON):
    cal_start = test_start - np.timedelta64(horizon, "D")
    d = pl.col("date")
    ts, cs = np.datetime64(test_start, "us"), np.datetime64(cal_start, "us")
    train = df.filter(d < cs)
    calib = df.filter((d >= cs) & (d < ts))
    test = df.filter((d >= ts) & (d < ts + np.timedelta64(horizon, "D")))
    return train, calib, test


def run_backtest(feats: pl.DataFrame, n_folds: int = 4, horizon: int = HORIZON, n_estimators: int = 300):
    last = feats["date"].max()
    rows, last_art = [], None
    for k in range(n_folds, 0, -1):
        test_start = last - np.timedelta64(k * horizon - 1, "D")
        train, calib, test = split_fold(feats, test_start, horizon)
        m = QuantileForecaster(n_estimators=n_estimators).fit(train, calib)
        y = test["sales"].to_numpy()
        conf, raw = m.predict(test, True), m.predict(test, False)
        base = seasonal_naive(test)
        rows.append({
            "fold": n_folds - k + 1, "test_start": str(test_start)[:10], "n_test": len(y),
            "wape_model": round(wape(y, conf["pred"]), 4), "wape_naive": round(wape(y, base), 4),
            "coverage_raw": round(float(((y >= raw["lo"]) & (y <= raw["hi"])).mean()), 4),
            "coverage_cqr": round(float(((y >= conf["lo"]) & (y <= conf["hi"])).mean()), 4),
            "width_cqr_rel": round(float((conf["hi"] - conf["lo"]).sum() / max(y.sum(), 1e-9)), 4),
            "q_hat_log": round(m.q_hat, 4),
        })
        last_art = (m, test, conf)
    return rows, last_art
