"""
notebooks/05_backtesting.py
───────────────────────────
Phase 5 — Backtesting demo script.

Run from the project root:
    python notebooks/05_backtesting.py

What this script does:
  1. Integrates the ENTIRE pipeline (Phases 1 → 4).
  2. Pulls target weights across the simulated history.
  3. Uses `vectorbt` to run a true, event-driven backtest.
  4. Compares against Buy-and-Hold SPY.
"""

import os
import sys
import logging

import pandas as pd
import numpy as np

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s │ %(name)s │ %(message)s")

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.regime_detector import RegimeDetector
from src.portfolio_optimizer import PortfolioOptimizer
from src.backtester import Backtester
from src.config import HMM_MODEL_PATH, ETF_TICKERS

def _banner(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def main():
    print("\n" + "=" * 60)
    print("  Market Regime Detector — Phase 5: Backtesting")
    print("=" * 60)

    # ── 1. Pipeline execution ────────────────────────────────────────────────
    _banner("Step 1 │ Running Full Data & Regime Pipeline")
    
    loader = DataLoader()
    prices = loader.fetch_etf_data()
    print(f"   ETF Prices            : {prices.shape}")
    
    if not os.path.exists(HMM_MODEL_PATH):
        print("   ❌ Model not found. Run Phase 3 (`notebooks/03_regime_detection.py`) first.")
        return
        
    combined = loader.get_combined_data()
    
    # Check if macro data is missing due to no FRED API key
    if "vix" not in combined.columns or combined["vix"].isna().all():
        print("   ⚠️  Macro data missing (no FRED key). Appending synthetic macro data for demo.")
        n = len(combined)
        rng = np.random.default_rng(42)
        combined["vix"] = rng.uniform(10, 40, n)
        combined["yield_10y"] = rng.uniform(1.5, 4.5, n)
        combined["yield_2y"] = rng.uniform(0.5, 4.0, n)
        combined["cpi"] = 272.0 + np.cumsum(rng.uniform(0.0, 0.3, n))
        combined["unemployment"] = rng.uniform(3.5, 10.0, n)
        combined["yield_spread"] = combined["yield_10y"] - combined["yield_2y"]
        
    fe = FeatureEngineer()
    X_raw = fe.compute_features(combined)
    X_norm = fe.fit_transform(X_raw)
    
    rd = RegimeDetector.load(HMM_MODEL_PATH)
    regimes = rd.predict(X_norm)
    
    # Restrict processing to the subset where we have HMM regimes configured.
    prices_aligned = prices.loc[regimes.index]
    # Restrict to strictly defined universe
    valid_tickers = [c for c in prices_aligned.columns if str(c).upper() in ETF_TICKERS]
    prices_aligned = prices_aligned[valid_tickers]
    
    print(f"   Detected Regimes      : {regimes.shape}")
    
    # ── 2. Weight Generation ─────────────────────────────────────────────────
    _banner("Step 2 │ Generating Target Weights (Phase 4)")
    
    po = PortfolioOptimizer()
    
    # In a full simulation, this step takes time. So we'll limit the backtest to the last 4 years
    # roughly 1000 days to keep the demo quick and responsive.
    SIM_DAYS = 1000
    if len(prices_aligned) > SIM_DAYS + po.lookback_days:
        sim_prices = prices_aligned.iloc[-SIM_DAYS - po.lookback_days:]
        sim_regimes = regimes.iloc[-SIM_DAYS - po.lookback_days:]
    else:
        sim_prices = prices_aligned
        sim_regimes = regimes
        
    weights = po.get_historical_weights(sim_prices, sim_regimes)
    
    # Wait for the lookback window to fill up, we only trade the final SIM_DAYS
    weights = weights.dropna().iloc[-SIM_DAYS:]
    sim_prices = sim_prices.loc[weights.index]

    print(f"   Calculated Weights    : {weights.shape[0]} days")
    
    # ── 3. Backtesting ───────────────────────────────────────────────────────
    _banner("Step 3 │ Running VectorBT Simulation (Phase 5)")
    
    bt = Backtester()
    print(f"   Transaction Cost      : {bt.transaction_cost * 100:.3f}% per trade")
    
    bt.run_backtest(sim_prices, weights)
    bt.run_benchmark(sim_prices)
    
    metrics = bt.get_metrics()
    
    # ── 4. Results ───────────────────────────────────────────────────────────
    _banner("Summary Metrics")
    print(metrics.round(2).to_string())
    
    # Print advice
    print("\n   To view the interactive equity curve chart:")
    print("   Run this script in an interactive python shell or remove `show=False` ")
    print("   from `bt.plot_equity_curve()`. (Note: Requires a browser window).")
    
    print("\n" + "=" * 60)
    print("  ✅ Phase 5 complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
