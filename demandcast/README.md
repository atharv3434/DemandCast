# DemandCast: Probabilistic Demand Forecasting

Forecast 28 days ahead for many products at once, with calibrated uncertainty,
validated by rolling-origin backtests.

## Techniques
| Idea | Implementation |
|---|---|
| Fast columnar feature engineering | **Polars** window expressions (`.over("series_id")`) |
| Leak-free direct multi-horizon forecasting | Every lag/rolling feature shifted by at least 28 days, so no recursive prediction is needed |
| Global model | One **LightGBM** model across all series, with series code, calendar, promo and holiday features |
| Probabilistic forecasts | LightGBM quantile regression (10% / 50% / 90%) on `log1p(sales)` |
| Calibrated intervals | **Conformalized Quantile Regression (CQR)** using a held-out calibration window |
| Honest evaluation | Rolling-origin backtest (train, calibration, test) and a seasonal-naive baseline |

## Run
```bash
pip install -r requirements.txt
export PYTHONPATH=src
make run    # writes outputs/metrics.json, forecast.png, coverage.png, demand.parquet
make test
```
Options: `python -m demandcast.run --series 100 --folds 6 --trees 500`

## Results (40 series, 4 folds of 28 days, synthetic data)
| Metric | Seasonal naive | DemandCast |
|---|---|---|
| WAPE | 0.386 | **0.280** (27.6% lower) |
| 80% interval coverage | n/a | raw 0.755, with CQR 0.790 |

CQR moved average coverage closer to the 80% target. Per-fold coverage still varies
(0.74 to 0.85), as expected when the calibration window is only 28 days and may not
represent the next window. Larger calibration windows or adaptive conformal
methods are the natural next step.

## Extend
- Add Optuna tuning for the quantile models
- Adaptive Conformal Inference for drifting data
- Hierarchical reconciliation (store, region, total)
- Swap in real data such as M5 or Favorita (needs `series_id`, `date`, `sales`)
- Serve forecasts with FastAPI or schedule retraining
