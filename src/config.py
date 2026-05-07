"""
config.py
─────────
Central configuration for the Market Regime Detector + Portfolio Optimizer.
All tickers, FRED series IDs, dates, and parameters live here.
Change this file to adjust the universe — nothing else needs to change.
"""

from datetime import date

# ─────────────────────────────────────────────────────────────────────────────
# 1. ETF Universe
# ─────────────────────────────────────────────────────────────────────────────
ETF_TICKERS: list[str] = [
    "SPY",   # SPDR S&P 500 ETF       — core US equity
    "QQQ",   # Invesco Nasdaq 100     — growth / risk-on
    "IWM",   # iShares Russell 2000   — small-cap equity
    "GLD",   # SPDR Gold Shares       — safe haven / inflation hedge
    "TLT",   # iShares 20+ Yr Treasury— long bonds / bear hedge
    "LQD",   # iShares IG Corp Bonds  — credit / yield
    "VNQ",   # Vanguard Real Estate   — real estate / inflation
    "USO",   # United States Oil Fund — commodity / macro signal
    "EFA",   # iShares MSCI EAFE      — international developed markets
    "EEM",   # iShares MSCI EM        — emerging markets
]

# ─────────────────────────────────────────────────────────────────────────────
# 2. FRED Macro Series
# ─────────────────────────────────────────────────────────────────────────────
# Keys are human-readable names used as DataFrame column names.
# Values are official FRED series IDs.
FRED_SERIES: dict[str, str] = {
    "vix":            "VIXCLS",      # CBOE VIX (daily)
    "yield_10y":      "DGS10",       # 10-Year Treasury Constant Maturity (daily)
    "yield_2y":       "DGS2",        # 2-Year Treasury Constant Maturity (daily)
    "cpi":            "CPIAUCSL",    # CPI All Urban Consumers (monthly → resampled daily)
    "unemployment":   "UNRATE",      # Unemployment Rate (monthly → resampled daily)
    "fed_funds":      "FEDFUNDS",    # Effective Fed Funds Rate (monthly)
}

# Computed after fetching (not a FRED series ID):
# "yield_spread" = yield_10y - yield_2y

# ─────────────────────────────────────────────────────────────────────────────
# 3. Date Range — 15+ Years of History
# ─────────────────────────────────────────────────────────────────────────────
START_DATE: str = "2005-01-01"  # Captures pre-GFC baseline
END_DATE:   str = date.today().strftime("%Y-%m-%d")  # Always up to today

# ─────────────────────────────────────────────────────────────────────────────
# 4. Cache / Storage Paths
# ─────────────────────────────────────────────────────────────────────────────
import os
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, "data")
RAW_DIR   = os.path.join(DATA_DIR, "raw")
PROC_DIR  = os.path.join(DATA_DIR, "processed")

ETF_CACHE_PATH   = os.path.join(RAW_DIR,  "etf_prices.parquet")
MACRO_CACHE_PATH = os.path.join(RAW_DIR,  "macro_data.parquet")
COMBINED_PATH    = os.path.join(PROC_DIR, "combined_dataset.parquet")

MODELS_DIR     = os.path.join(BASE_DIR, "models")
HMM_MODEL_PATH = os.path.join(MODELS_DIR, "regime_hmm.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# 5. HMM / Model Parameters (used in Phase 3 — defined here for central access)
# ─────────────────────────────────────────────────────────────────────────────
N_REGIMES:         int = 3     # Bull / Bear / Sideways
REFIT_WINDOW_DAYS: int = 252   # Rolling refit every 252 trading days
MIN_REGIME_HOLD:   int = 5     # Min days before confirming a regime change

# ─────────────────────────────────────────────────────────────────────────────
# 6. Portfolio Constraints (Phase 4)
# ─────────────────────────────────────────────────────────────────────────────
MAX_SINGLE_WEIGHT: float = 0.30   # No single asset > 30%
MIN_SINGLE_WEIGHT: float = 0.05   # No single asset < 5%

# ─────────────────────────────────────────────────────────────────────────────
# 7. Backtest Parameters (Phase 5)
# ─────────────────────────────────────────────────────────────────────────────
TRANSACTION_COST: float = 0.000   # 0.0% per trade for pure mathematical edge (adjustable)
BENCHMARK_TICKERS: list[str] = ["SPY"]

# ─────────────────────────────────────────────────────────────────────────────
# 8. Regime Labels (assigned AFTER HMM training — states are unlabeled by default)
# ─────────────────────────────────────────────────────────────────────────────
REGIME_NAMES: dict[int, str] = {
    0: "Bull",
    1: "Bear",
    2: "Sideways",
}

REGIME_COLORS: dict[str, str] = {
    "Bull":     "#22c55e",   # green
    "Bear":     "#ef4444",   # red
    "Sideways": "#f59e0b",   # amber
}
