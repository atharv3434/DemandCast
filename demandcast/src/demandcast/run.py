"""CLI: generate data, backtest, write metrics + plots."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import make_demand
from .features import build_features, FEATURES
from .backtest import run_backtest

OUT = Path(__file__).resolve().parents[2] / "outputs"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", type=int, default=40)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--trees", type=int, default=300)
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)

    raw = make_demand(a.series)
    raw.write_parquet(OUT / "demand.parquet")
    feats = build_features(raw)
    rows, (model, test, conf) = run_backtest(feats, a.folds, n_estimators=a.trees)

    avg = lambda k: round(float(np.mean([r[k] for r in rows])), 4)
    summary = {"folds": rows, "mean": {k: avg(k) for k in
               ("wape_model", "wape_naive", "coverage_raw", "coverage_cqr", "width_cqr_rel")}}
    summary["mean"]["wape_improvement_pct"] = round(
        100 * (1 - summary["mean"]["wape_model"] / summary["mean"]["wape_naive"]), 1)
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))

    # Plot 1: forecast with conformal band for the busiest series in the last fold
    t = test.with_columns(pl.Series("pred", conf["pred"]), pl.Series("lo", conf["lo"]), pl.Series("hi", conf["hi"]))
    sid = t.group_by("series_id").agg(pl.col("sales").sum()).sort("sales", descending=True)["series_id"][0]
    ts = t.filter(pl.col("series_id") == sid).sort("date")
    hist = raw.filter((pl.col("series_id") == sid) & (pl.col("date") < ts["date"].min())).sort("date").tail(60)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(hist["date"], hist["sales"], color="#9ca3af", label="history")
    ax.fill_between(ts["date"], ts["lo"], ts["hi"], color="#6366f1", alpha=.25, label="80% interval (CQR)")
    ax.plot(ts["date"], ts["pred"], color="#4f46e5", label="forecast")
    ax.scatter(ts["date"], ts["sales"], s=14, color="#111827", label="actual", zorder=3)
    ax.set_title(f"28-day forecast, series {sid}"); ax.legend(ncol=4, fontsize=8, loc="upper left")
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT / "forecast.png", dpi=140); plt.close(fig)

    # Plot 2: coverage raw vs conformal per fold
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(rows)); w = .38
    ax.bar(x - w / 2, [r["coverage_raw"] for r in rows], w, label="raw quantile", color="#fca5a5")
    ax.bar(x + w / 2, [r["coverage_cqr"] for r in rows], w, label="after CQR", color="#4f46e5")
    ax.axhline(0.8, color="k", ls="--", lw=1); ax.set_xticks(x, [f"fold {r['fold']}" for r in rows])
    ax.set_ylim(0.5, 1); ax.set_ylabel("empirical coverage"); ax.set_title("Target coverage: 80%")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "coverage.png", dpi=140); plt.close(fig)

    print(json.dumps(summary["mean"], indent=2))
    return summary


if __name__ == "__main__":
    main()
