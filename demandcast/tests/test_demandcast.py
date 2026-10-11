import numpy as np
from demandcast.data import make_demand
from demandcast.features import build_features, HORIZON
from demandcast.backtest import run_backtest, wape


def test_no_leakage_all_lags_at_least_horizon():
    raw = make_demand(3, days=500)
    f = build_features(raw)
    one = f.filter(f["series_id"] == "S000").sort("date")
    full = raw.filter(raw["series_id"] == "S000").sort("date")
    row = one[200]
    d = row["date"][0]
    idx = int(np.where(full["date"].to_numpy() == np.datetime64(d, "us"))[0][0])
    assert row["lag_28"][0] == full["sales"][idx - 28]
    assert row["lag_56"][0] == full["sales"][idx - 56]


def test_backtest_beats_naive_and_covers():
    raw = make_demand(10, days=700)
    rows, _ = run_backtest(build_features(raw), n_folds=2, n_estimators=80)
    assert np.mean([r["wape_model"] for r in rows]) < np.mean([r["wape_naive"] for r in rows])
    assert all(0.65 <= r["coverage_cqr"] <= 0.95 for r in rows)


def test_wape():
    assert wape(np.array([10., 10.]), np.array([8., 12.])) == 0.2
