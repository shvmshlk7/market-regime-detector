"""
notebooks/03_regime_detection.py
──────────────────────────────────
Phase 3 — HMM Market Regime Detection demo script.

Run from the project root:
    python notebooks/03_regime_detection.py

What this script does:
  1. Load (or re-compute) feature matrix from Phase 2 cache
  2. Fit GaussianHMM on 80% training split
  3. Predict regimes on full history
  4. Display regime summary statistics
  5. Save trained model to models/regime_hmm.pkl
  6. Demonstrate model round-trip via load()

No real API calls are needed if the parquet cache already exists.
If cache is absent, the script generates a clean synthetic dataset.
"""

import os
import sys
import logging

import numpy as np
import pandas as pd

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level  = logging.WARNING,   # Suppress verbose HMM debug logs in demo
    format = "%(levelname)s │ %(name)s │ %(message)s",
)

from src.feature_engineer import FeatureEngineer
from src.regime_detector  import RegimeDetector
from src.config import (
    COMBINED_PATH,
    HMM_MODEL_PATH,
    N_REGIMES,
    MIN_REGIME_HOLD,
    REGIME_NAMES,
    REGIME_COLORS,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def _load_or_generate_data() -> pd.DataFrame:
    """
    Try to load the combined dataset from Phase 1 cache.
    Falls back to a clean synthetic dataset if cache is absent.
    """
    if os.path.exists(COMBINED_PATH):
        print(f"✅ Loading cached data from: {COMBINED_PATH}")
        data = pd.read_parquet(COMBINED_PATH)
        # Lower-case column names (DataLoader may store uppercase tickers)
        data.columns = data.columns.str.lower()
        return data

    print("⚠️  No cache found — generating synthetic data (realistic GBM).")
    n   = 1000
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2005-01-03", periods=n)

    def _gbm(S0=100.0, mu=0.0001, sigma=0.012):
        r = rng.normal(mu, sigma, n)
        return S0 * np.exp(np.cumsum(r))

    data = pd.DataFrame({
        "spy":          _gbm(100.0),
        "qqq":          _gbm(200.0),
        "tlt":          _gbm(90.0,  mu=-0.00005, sigma=0.010),
        "uso":          _gbm(30.0,  mu=0.0,      sigma=0.020),
        "gld":          _gbm(150.0, mu=0.00005,  sigma=0.008),
        "vix":          rng.uniform(10, 40, n),
        "yield_10y":    rng.uniform(1.5, 4.5, n),
        "yield_2y":     rng.uniform(0.5, 4.0, n),
        "cpi":          272.0 + np.cumsum(rng.uniform(0.0, 0.3, n)),
        "unemployment": rng.uniform(3.5, 10.0, n),
    }, index=dates)
    data.index.name  = "date"
    data["yield_spread"] = data["yield_10y"] - data["yield_2y"]
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  Market Regime Detector — Phase 3: HMM Regime Detection")
    print("=" * 60)

    # ── Step 1: Load data ────────────────────────────────────────────────────
    _banner("Step 1 │ Load Data")
    data = _load_or_generate_data()
    print(f"   Rows      : {len(data):,}")
    print(f"   Date range: {data.index[0].date()} → {data.index[-1].date()}")
    print(f"   Columns   : {list(data.columns)}")

    # ── Step 2: Feature Engineering (Phase 2) ───────────────────────────────
    _banner("Step 2 │ Feature Engineering")
    fe    = FeatureEngineer()
    X_raw = fe.compute_features(data)
    X_norm = fe.fit_transform(X_raw)
    print(f"   Feature matrix shape : {X_norm.shape}")
    print(f"   Features             : {list(X_norm.columns)}")
    print(f"   Date range           : {X_norm.index[0].date()} → {X_norm.index[-1].date()}")

    # ── Step 3: Train/test split (80 / 20) ──────────────────────────────────
    _banner("Step 3 │ Train / Test Split  (80% / 20%)")
    split      = int(len(X_norm) * 0.80)
    X_train    = X_norm.iloc[:split]
    X_test     = X_norm.iloc[split:]
    X_raw_test = X_raw.iloc[split:]
    print(f"   Training rows : {len(X_train):,} ({X_train.index[0].date()} → {X_train.index[-1].date()})")
    print(f"   Test rows     : {len(X_test):,}  ({X_test.index[0].date()} → {X_test.index[-1].date()})")

    # ── Step 4: Fit HMM ──────────────────────────────────────────────────────
    _banner(f"Step 4 │ Fit Gaussian HMM  (n_regimes={N_REGIMES})")
    rd = RegimeDetector(
        n_regimes       = N_REGIMES,
        n_iter          = 200,
        covariance_type = "full",
        random_state    = 42,
        min_regime_hold = MIN_REGIME_HOLD,
    )
    rd.fit(X_train)
    print(rd.summary())

    # ── Step 5: Predict on full history ─────────────────────────────────────
    _banner("Step 5 │ Predict Regimes (Full History)")
    regimes_full = rd.predict(X_norm)
    counts       = regimes_full.value_counts()
    print(f"\n   Regime distribution:")
    for regime, cnt in counts.items():
        pct   = cnt / len(regimes_full) * 100
        color = REGIME_COLORS.get(regime, "")
        bar   = "█" * int(pct / 2)
        print(f"     {regime:<10}: {cnt:>4} days ({pct:5.1f}%)  {bar}")

    # ── Step 6: Regime probabilities (last 10 rows) ─────────────────────────
    _banner("Step 6 │ Regime Probabilities — Last 10 Trading Days")
    proba = rd.predict_proba(X_norm)
    print(proba.tail(10).round(3).to_string())

    # ── Step 7: Regime statistics ────────────────────────────────────────────
    _banner("Step 7 │ Regime Statistics (raw features)")
    stats  = rd.get_regime_stats(X_norm, X_raw)
    meta   = stats["meta"] if "meta" in stats.columns.get_level_values(0) else None
    if meta is not None:
        print(f"\n   {'Regime':<12} {'Count':>6}  {'% of Time':>10}")
        print(f"   {'──────':<12} {'─────':>6}  {'─────────':>10}")
        for regime in meta.index:
            cnt = int(meta.loc[regime, "count"])
            pct = float(meta.loc[regime, "pct"])
            print(f"   {regime:<12} {cnt:>6}  {pct:>9.1f}%")

    # Per-regime key feature means
    key_features = ["spy_log_ret", "spy_vol_20d", "vix_level"]
    available    = [f for f in key_features if (f, "mean") in stats.columns]
    if available:
        print(f"\n   Mean feature values per regime:")
        header = f"   {'Regime':<12}" + "".join(f"  {f:<18}" for f in available)
        print(header)
        print("   " + "─" * (len(header) - 3))
        for regime in stats.index:
            row = f"   {regime:<12}"
            for f in available:
                val = stats.loc[regime, (f, "mean")]
                row += f"  {val:>18.4f}"
            print(row)

    # ── Step 8: Out-of-sample prediction ────────────────────────────────────
    _banner("Step 8 │ Out-of-Sample Prediction (test set)")
    regimes_test = rd.predict(X_test)
    oos_counts   = regimes_test.value_counts()
    print(f"   Test set regime distribution:")
    for regime, cnt in oos_counts.items():
        pct = cnt / len(regimes_test) * 100
        print(f"     {regime:<10}: {cnt:>4} ({pct:.1f}%)")

    # ── Step 9: Save model ───────────────────────────────────────────────────
    _banner("Step 9 │ Save Model")
    rd.save(HMM_MODEL_PATH)
    print(f"   ✅ Model saved → {HMM_MODEL_PATH}")

    # ── Step 10: Load and verify round-trip ──────────────────────────────────
    _banner("Step 10 │ Load Model (round-trip verification)")
    rd2      = RegimeDetector.load(HMM_MODEL_PATH)
    regimes2 = rd2.predict(X_norm)
    match    = (regimes_full == regimes2).all()
    print(f"   Model loaded ← {HMM_MODEL_PATH}")
    print(f"   Predictions identical: {'✅ YES' if match else '❌ NO — mismatch!'}")

    print("\n" + "=" * 60)
    print("  ✅ Phase 3 complete!")
    print(f"     Model       : {HMM_MODEL_PATH}")
    print(f"     Data range  : {X_norm.index[0].date()} → {X_norm.index[-1].date()}")
    print(f"     Total days  : {len(regimes_full):,}")
    print(f"     Label map   : {rd._label_map}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
