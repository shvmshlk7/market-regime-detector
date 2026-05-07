"""
notebooks/04_portfolio_optimization.py
──────────────────────────────────────
Phase 4 — Portfolio Optimization demo script.

Run from the project root:
    python notebooks/04_portfolio_optimization.py

What this script does:
  1. Load underlying price data and predicted regimes (generates them if missing)
  2. Instantiate the PortfolioOptimizer
  3. Generate optimal Target Portfolios for current Bull, Bear, and Sideways regimes
  4. Compare the allocations visually in terminal output
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
from src.config import HMM_MODEL_PATH, ETF_TICKERS

def _banner(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")

def _bar_chart(weight: float, width: int = 20) -> str:
    """Returns a text-based progress bar for weight allocations."""
    filled = int(round(weight * width))
    empty = width - filled
    return "█" * filled + "░" * empty

def main():
    print("\n" + "=" * 60)
    print("  Market Regime Detector — Phase 4: Portfolio Optimization")
    print("=" * 60)

    # ── 1. Fetching Data ──────────────────────────────────────────────────
    _banner("Step 1 │ Loading Price Data and Regimes")
    
    loader = DataLoader()
    prices = loader.fetch_etf_data()
    print(f"   ETF Prices loaded     : {prices.shape}")
    
    # Needs regimes to demonstrate. We'll load the model and predict
    if not os.path.exists(HMM_MODEL_PATH):
        print("   ⚠️  No trained HMM model found. Run Phase 3 first.")
        print("   Generating synthetic regimes for demonstration...")
        dates = prices.index
        n = len(dates)
        synth_regimes = np.array(["Bull"] * (n // 3) + ["Bear"] * (n // 3) + ["Sideways"] * (n - 2 * (n // 3)))
        regimes = pd.Series(synth_regimes, index=dates)
    else:
        combined = loader.get_combined_data()
        fe = FeatureEngineer()
        X_raw = fe.compute_features(combined)
        X_norm = fe.fit_transform(X_raw)
        
        rd = RegimeDetector.load(HMM_MODEL_PATH)
        # Note: Regimes length will be shorter than prices due to warmup period.
        # Align prices
        regimes = rd.predict(X_norm)
        prices = prices.loc[regimes.index]
        
    print(f"   Regimes loaded        : {regimes.shape}")
    
    # Keep only the columns configured
    available_tickers = [c for c in prices.columns if str(c).upper() in ETF_TICKERS]
    prices = prices[available_tickers]

    # ── 2. Create Optimizer ────────────────────────────────────────────────
    _banner("Step 2 │ Initializing Portfolio Optimizer")
    po = PortfolioOptimizer()
    print(f"   Min weight bound      : {po.min_weight * 100:.1f}%")
    print(f"   Max weight bound      : {po.max_weight * 100:.1f}%")
    print(f"   Lookback Days         : {po.lookback_days} days")
    
    # ── 3. Optimize Target Portfolios ──────────────────────────────────────
    _banner("Step 3 │ Generating Regime-Target Portfolios")
    
    # We use the trailing 252 days of prices
    recent_prices = prices.iloc[-252:]
    
    target_portfolios = {}
    for r in ["Bull", "Bear", "Sideways"]:
        w = po.optimize(recent_prices, r)
        target_portfolios[r] = w
        
    # Formatting output target portfolios
    for r in ["Bull", "Bear", "Sideways"]:
        print(f"\n   Target {r} Portfolio Allocation:")
        print(f"   {'Ticker':<8} | {'Weight':>8} | {'Allocation Bar'}")
        print("   " + "-" * 42)
        
        sorted_weights = dict(sorted(target_portfolios[r].items(), key=lambda item: item[1], reverse=True))
        
        for ticker, weight in sorted_weights.items():
            pct = weight * 100
            if pct > 0.1:  # Only show meaningful weights
                print(f"   {str(ticker).upper():<8} | {pct:>7.2f}% | {_bar_chart(weight)}")
                
    # ── 4. Historical Back-Fill ───────────────────────────────────────────
    _banner("Step 4 │ Calculating Historical Rolling Weights (Summary)")
    print("   (This takes a few seconds as it calculates weights rolling forward...)")
    
    # For demo speed, just calculate last 2 years
    start_idx = max(0, len(prices) - 504)
    hist_weights = po.get_historical_weights(prices.iloc[start_idx:], regimes.iloc[start_idx:])
    
    print(f"\n   Historical Weights shape: {hist_weights.shape}")
    print("\n   Sample Recent Turnovers (last 3 rebalance dates):")
    
    # Identify rebalance dates visually
    rebalance_dates = hist_weights.diff().abs().sum(axis=1)
    rebalance_dates = rebalance_dates[rebalance_dates > 0.01].index
    
    if len(rebalance_dates) > 0:
        for d in rebalance_dates[-3:]:
            reg = regimes.loc[d]
            print(f"     {d.date()} -> Switched to {reg}")
    else:
        print("     No regime switches in recent history.")
        
    print("\n" + "=" * 60)
    print("  ✅ Phase 4 complete!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
