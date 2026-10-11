"""Global LightGBM quantile models + Conformalized Quantile Regression (CQR) + baseline."""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from .features import FEATURES

ALPHA = 0.2  # 80% prediction interval


class QuantileForecaster:
    """Trains on log1p(sales). Quantiles are invariant under monotone transforms,
    so back-transforming preserves the interval semantics while stabilising scale across series."""

    def __init__(self, alpha: float = ALPHA, n_estimators: int = 300, seed: int = 0):
        self.alpha, self.seed, self.n = alpha, seed, n_estimators
        self.models: dict[str, lgb.LGBMRegressor] = {}
        self.q_hat = 0.0

    def _mk(self, a: float) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(objective="quantile", alpha=a, n_estimators=self.n, learning_rate=0.05,
                                 num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, random_state=self.seed, verbosity=-1)

    @staticmethod
    def _X(df: pl.DataFrame) -> pd.DataFrame:
        return df.select(FEATURES).to_pandas()

    def fit(self, train: pl.DataFrame, calib: pl.DataFrame) -> "QuantileForecaster":
        X, y = self._X(train), np.log1p(train["sales"].to_numpy())
        for name, a in (("lo", self.alpha / 2), ("mid", 0.5), ("hi", 1 - self.alpha / 2)):
            self.models[name] = self._mk(a).fit(X, y)
        # conformal correction computed on a held-out calibration window
        lo, _, hi = self._raw(calib)
        yc = np.log1p(calib["sales"].to_numpy())
        scores = np.maximum(lo - yc, yc - hi)
        n = len(scores)
        level = min(1.0, math.ceil((n + 1) * (1 - self.alpha)) / n)
        self.q_hat = float(np.quantile(scores, level))
        return self

    def _raw(self, df: pl.DataFrame):
        X = self._X(df)
        lo, mid, hi = (self.models[k].predict(X) for k in ("lo", "mid", "hi"))
        lo, hi = np.minimum(lo, mid), np.maximum(hi, mid)  # fix quantile crossing
        return lo, mid, hi

    def predict(self, df: pl.DataFrame, conformal: bool = True) -> dict[str, np.ndarray]:
        lo, mid, hi = self._raw(df)
        q = self.q_hat if conformal else 0.0
        f = lambda z: np.clip(np.expm1(z), 0, None)
        return {"lo": f(lo - q), "pred": f(mid), "hi": f(hi + q)}


def seasonal_naive(df: pl.DataFrame) -> np.ndarray:
    """Same weekday 4 weeks ago (falls back to the 4-week same-weekday mean)."""
    return df.select(pl.coalesce("lag_28", "same_dow_mean")).to_numpy().ravel()
