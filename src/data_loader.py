"""
data_loader.py
──────────────
DataLoader class for the Market Regime Detector + Portfolio Optimizer.

Responsibilities:
  1. Fetch OHLCV data for the ETF universe via yfinance (20+ years)
  2. Fetch macro indicators from FRED (VIX, yield curve, CPI, unemployment)
  3. Cache everything to Parquet — no repeated API calls between runs
  4. Merge ETF + macro into one aligned DataFrame (business days)
  5. Handle missing data, forward-fill, and validate data quality
  6. Expose a clean combined_data property ready for FeatureEngineer (Phase 2)

Usage:
    from src.data_loader import DataLoader

    loader = DataLoader()
    etf    = loader.fetch_etf_data()         # Returns adj-close prices
    macro  = loader.fetch_macro_data()        # Returns VIX, spread, CPI, etc.
    data   = loader.get_combined_data()       # Merged & cleaned, ready for ML
    report = loader.validate_data(data)       # Data quality report
"""

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

# Optional: FRED API — gracefully degrade if key not provided
try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False

from src.config import (
    ETF_TICKERS,
    FRED_SERIES,
    START_DATE,
    END_DATE,
    ETF_CACHE_PATH,
    MACRO_CACHE_PATH,
    COMBINED_PATH,
    RAW_DIR,
    PROC_DIR,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup — INFO goes to console, DEBUG available when needed
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Fetches, caches, and validates market + macro data for the project.

    Parameters
    ----------
    start_date : str
        Start of the data window (YYYY-MM-DD). Defaults to config.START_DATE.
    end_date : str
        End of the data window (YYYY-MM-DD). Defaults to today.
    tickers : list[str] | None
        ETF tickers to fetch. Defaults to config.ETF_TICKERS.
    fred_api_key : str | None
        FRED API key. Falls back to FRED_API_KEY environment variable.
        If neither is provided, macro data is fetched with reduced coverage.
    cache_dir_raw : str
        Directory for raw parquet cache files.
    cache_dir_processed : str
        Directory for processed/combined parquet file.
    """

    def __init__(
        self,
        start_date: str = START_DATE,
        end_date:   str = END_DATE,
        tickers:    Optional[list[str]] = None,
        fred_api_key: Optional[str] = None,
        cache_dir_raw: str = RAW_DIR,
        cache_dir_processed: str = PROC_DIR,
    ):
        self.start_date = start_date
        self.end_date   = end_date
        self.tickers    = tickers or ETF_TICKERS

        # ── FRED API key ────────────────────────────────────────────────────
        self.fred_api_key = (
            fred_api_key
            or os.environ.get("FRED_API_KEY")
        )
        self._fred: Optional["Fred"] = None  # Lazy-loaded

        # ── Cache paths ─────────────────────────────────────────────────────
        self.etf_cache_path   = ETF_CACHE_PATH
        self.macro_cache_path = MACRO_CACHE_PATH
        self.combined_path    = COMBINED_PATH

        # Ensure directories exist
        os.makedirs(cache_dir_raw,       exist_ok=True)
        os.makedirs(cache_dir_processed, exist_ok=True)

        logger.info(
            f"DataLoader initialized | "
            f"tickers={len(self.tickers)} | "
            f"window={self.start_date} → {self.end_date} | "
            f"FRED={'✓' if self.fred_api_key else '✗ (no key)'}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FRED API — lazy init
    # ─────────────────────────────────────────────────────────────────────────
    @property
    def fred(self) -> Optional["Fred"]:
        """Lazy-load FRED client. Returns None if key unavailable."""
        if self._fred is None:
            if not FRED_AVAILABLE:
                logger.warning("fredapi not installed. Run: pip install fredapi")
                return None
            if not self.fred_api_key:
                logger.warning(
                    "No FRED API key found. Set FRED_API_KEY in your .env file. "
                    "Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
                )
                return None
            self._fred = Fred(api_key=self.fred_api_key)
            logger.info("FRED API client initialized.")
        return self._fred

    # ─────────────────────────────────────────────────────────────────────────
    # 1. ETF Data
    # ─────────────────────────────────────────────────────────────────────────
    def fetch_etf_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch adjusted-close prices for all ETF tickers via yfinance.

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex (business days)
            Columns: ticker symbols (e.g. SPY, QQQ, ...)
            Values: adjusted closing prices

        Notes
        -----
        - auto_adjust=True handles splits, dividends automatically (true total return)
        - Results cached to parquet; subsequent calls load from cache in <1s
        - Corporate actions and splits handled automatically by yfinance
        """
        if not force_refresh and os.path.exists(self.etf_cache_path):
            logger.info(f"Loading ETF data from cache: {self.etf_cache_path}")
            df = pd.read_parquet(self.etf_cache_path)
            logger.info(f"Cache loaded: {df.shape[0]} rows × {df.shape[1]} tickers")
            return df

        logger.info(f"Fetching ETF data from yfinance: {self.tickers}")
        logger.info(f"Date range: {self.start_date} → {self.end_date}")

        raw = yf.download(
            tickers=self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,      # Adjust for splits + dividends (total return)
            progress=True,
            group_by="column",     # Fields at Level 0, tickers at Level 1 — consistent
            threads=True,          # Parallel downloads
        )

        # ── Extract adjusted close prices ────────────────────────────────────
        # yfinance 1.x with group_by='column': MultiIndex(level0=fields, level1=tickers)
        # yfinance with single ticker: MultiIndex(level0=fields, level1=[ticker]) or flat
        adj_close = self._extract_close(raw)

        adj_close.index = pd.to_datetime(adj_close.index)
        adj_close.index.name = "date"

        # ── Handle missing data ──────────────────────────────────────────────
        adj_close = self._clean_price_data(adj_close)

        # ── Cache to parquet ─────────────────────────────────────────────────
        adj_close.to_parquet(self.etf_cache_path, index=True)
        logger.info(
            f"ETF data saved to cache: {adj_close.shape[0]} rows × "
            f"{adj_close.shape[1]} tickers → {self.etf_cache_path}"
        )

        return adj_close

    def _extract_close(self, raw: pd.DataFrame) -> pd.DataFrame:
        """
        Robustly extract Close prices from yfinance output regardless of version.

        yfinance 1.x (group_by='column'):
            MultiIndex — Level 0 = fields (Close/Open/…), Level 1 = tickers
            → raw["Close"] gives a DataFrame with ticker columns  ✓

        yfinance legacy / single-ticker:
            Flat columns ["Open", "High", "Low", "Close", "Volume"]
            → raw[["Close"]] then rename
        """
        if isinstance(raw.columns, pd.MultiIndex):
            lvl0 = raw.columns.get_level_values(0).unique().tolist()
            lvl1 = raw.columns.get_level_values(1).unique().tolist()

            if "Close" in lvl0:
                # group_by='column' layout: (field, ticker) — expected path
                adj_close = raw["Close"].copy()
                logger.debug("yfinance: MultiIndex (field, ticker) — Close extraction OK")
            elif "Close" in lvl1:
                # group_by='ticker' layout: (ticker, field) — swap and extract
                logger.warning(
                    "yfinance returned (ticker, field) MultiIndex. "
                    "Swapping levels to extract Close prices."
                )
                raw_swapped = raw.swaplevel(0, 1, axis=1)
                raw_swapped.columns.names = ["field", "ticker"]
                adj_close = raw_swapped["Close"].copy()
            else:
                raise ValueError(
                    f"Cannot extract 'Close' from yfinance output. "
                    f"Level 0: {lvl0}, Level 1: {lvl1}. "
                    f"Check yfinance version compatibility."
                )
        else:
            # Flat columns — single ticker
            if "Close" not in raw.columns:
                raise ValueError(
                    f"'Close' column not found. Available: {raw.columns.tolist()}"
                )
            adj_close = raw[["Close"]].copy()
            adj_close.columns = self.tickers[:1]  # Use the first (only) ticker name
            logger.debug("yfinance: flat columns — single-ticker path")

        return adj_close

    def _clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate price data:
        - Remove weekends (keep trading days only)
        - Forward-fill small gaps (up to 5 days — handles holidays)
        - Drop tickers with >10% missing data after forward-fill
        - Log data quality summary
        """
        original_shape = df.shape

        # Keep only rows where at least one ticker has data (removes weekends)
        df = df[df.notna().any(axis=1)]

        # Forward-fill gaps up to 5 trading days (handles public holidays)
        df = df.ffill(limit=5)

        # Report missing data PER TICKER
        missing_pct = df.isna().sum() / len(df) * 100
        bad_tickers = missing_pct[missing_pct > 10].index.tolist()

        if bad_tickers:
            logger.warning(
                f"Dropping tickers with >10% missing data: {bad_tickers}"
            )
            df = df.drop(columns=bad_tickers)

        # Log per-ticker missing summary
        for ticker in df.columns:
            pct = df[ticker].isna().sum() / len(df) * 100
            if pct > 0:
                logger.info(f"  {ticker}: {pct:.2f}% missing (after forward-fill)")

        logger.info(
            f"Price data cleaned: {original_shape} → {df.shape} "
            f"| date range: {df.index[0].date()} → {df.index[-1].date()}"
        )
        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Macro Data
    # ─────────────────────────────────────────────────────────────────────────
    def fetch_macro_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch macro indicators from FRED.

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex (business days, resampled/forward-filled)
            Columns: vix, yield_10y, yield_2y, yield_spread, cpi, unemployment, fed_funds

        Notes
        -----
        - Monthly series (CPI, UNRATE) are resampled to daily and forward-filled
        - yield_spread = yield_10y - yield_2y (computed here, not a FRED series)
        - VIX is daily; yield spreads are daily; CPI/unemployment are monthly
        """
        if not force_refresh and os.path.exists(self.macro_cache_path):
            logger.info(f"Loading macro data from cache: {self.macro_cache_path}")
            df = pd.read_parquet(self.macro_cache_path)
            logger.info(f"Macro cache loaded: {df.shape}")
            return df

        if self.fred is None:
            logger.warning(
                "FRED client unavailable — returning empty macro DataFrame. "
                "The model will still work but without macro features."
            )
            return pd.DataFrame()

        logger.info("Fetching macro indicators from FRED...")

        series_dict = {}
        for name, series_id in FRED_SERIES.items():
            try:
                logger.info(f"  Fetching FRED series: {series_id} ({name})")
                s = self.fred.get_series(
                    series_id,
                    observation_start=self.start_date,
                    observation_end=self.end_date,
                )
                s.name = name
                series_dict[name] = s
                logger.info(f"  ✓ {name}: {len(s)} observations")
            except Exception as e:
                logger.error(f"  ✗ Failed to fetch {series_id}: {e}")

        if not series_dict:
            logger.error("No FRED data fetched. Check your API key.")
            return pd.DataFrame()

        # ── Combine into one DataFrame ───────────────────────────────────────
        macro = pd.concat(series_dict.values(), axis=1)
        macro.index = pd.to_datetime(macro.index)
        macro.index.name = "date"

        # ── Compute yield spread ─────────────────────────────────────────────
        if "yield_10y" in macro.columns and "yield_2y" in macro.columns:
            macro["yield_spread"] = macro["yield_10y"] - macro["yield_2y"]
            logger.info("  ✓ yield_spread computed (10Y - 2Y)")

        # ── Resample to business days + forward-fill ─────────────────────────
        macro = self._resample_macro_to_daily(macro)

        # ── Cache ────────────────────────────────────────────────────────────
        macro.to_parquet(self.macro_cache_path, index=True)
        logger.info(f"Macro data saved to cache: {macro.shape} → {self.macro_cache_path}")

        return macro

    def _resample_macro_to_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample mixed-frequency macro data to business-day frequency.
        Monthly data (CPI, unemployment) is forward-filled.
        Daily data (VIX, yields) is kept as-is with gaps filled.
        """
        # Create a full business-day date range
        bday_index = pd.bdate_range(start=self.start_date, end=self.end_date)

        # Reindex to business days
        df = df.reindex(bday_index)
        df.index.name = "date"

        # Forward-fill: monthly series will propagate monthly values daily
        # Limit=30 prevents runaway fill at series boundaries
        df = df.ffill(limit=30)

        # Log missing summary
        missing = df.isna().sum()
        if missing.sum() > 0:
            logger.info(f"Macro missing after fill: {missing[missing > 0].to_dict()}")

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Combined Dataset
    # ─────────────────────────────────────────────────────────────────────────
    def get_combined_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Merge ETF price data and macro indicators into one aligned DataFrame.

        Returns
        -------
        pd.DataFrame
            Index: DatetimeIndex (business days where BOTH datasets have data)
            Columns: [spy, qqq, ..., vix, yield_spread, cpi, unemployment, ...]
            All NaN rows are dropped; columns are lowercased for consistency.

        This is the canonical dataset consumed by FeatureEngineer in Phase 2.
        """
        if not force_refresh and os.path.exists(self.combined_path):
            logger.info(f"Loading combined dataset from cache: {self.combined_path}")
            df = pd.read_parquet(self.combined_path)
            logger.info(f"Combined cache loaded: {df.shape}")
            return df

        logger.info("Building combined dataset (ETF + macro)...")

        # Fetch both datasets
        etf   = self.fetch_etf_data(force_refresh=force_refresh)
        macro = self.fetch_macro_data(force_refresh=force_refresh)

        # Lowercase column names for consistency
        etf.columns   = [c.lower() for c in etf.columns]

        if macro.empty:
            logger.warning("Macro data empty — combined dataset will be ETF-only.")
            combined = etf
        else:
            # Align on business-day index (inner join = only days both have data)
            combined = etf.join(macro, how="left")

            # Forward-fill macro on days ETF traded but FRED is missing (holidays)
            macro_cols = macro.columns.tolist()
            combined[macro_cols] = combined[macro_cols].ffill(limit=5)

        # Drop rows where ALL ETF prices are missing
        etf_cols = [c for c in combined.columns if c.lower() in [t.lower() for t in self.tickers]]
        combined = combined.dropna(subset=etf_cols, how="all")

        # Report final shape
        logger.info(
            f"Combined dataset: {combined.shape[0]} rows × {combined.shape[1]} columns | "
            f"ETF cols: {len(etf_cols)} | Macro cols: {combined.shape[1] - len(etf_cols)} | "
            f"Date range: {combined.index[0].date()} → {combined.index[-1].date()}"
        )

        # Cache
        combined.to_parquet(self.combined_path, index=True)
        logger.info(f"Combined dataset saved: {self.combined_path}")

        return combined

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Data Quality Validation
    # ─────────────────────────────────────────────────────────────────────────
    def validate_data(self, df: pd.DataFrame) -> dict:
        """
        Run data quality checks and return a structured report.

        Returns
        -------
        dict with keys:
          - shape: (rows, cols)
          - date_range: (start, end)
          - trading_days: int
          - years_of_history: float
          - missing_pct: dict[col -> pct missing]
          - zero_return_days: dict[col -> count of zero-return days]
          - suspicious_columns: list[col with >5% missing]
          - passed: bool (True if data quality is acceptable)
        """
        logger.info("Running data validation...")

        report = {}
        report["shape"]          = df.shape
        report["date_range"]     = (str(df.index[0].date()), str(df.index[-1].date()))
        report["trading_days"]   = len(df)
        report["years_of_history"] = round(len(df) / 252, 1)

        # Missing data per column
        missing_pct = (df.isna().sum() / len(df) * 100).round(2)
        report["missing_pct"] = missing_pct.to_dict()

        # Zero-return days (potential data errors)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        zero_return_days = {}
        for col in numeric_cols:
            returns = df[col].pct_change().dropna()
            zero_days = (returns == 0).sum()
            if zero_days > 10:
                zero_return_days[col] = int(zero_days)
        report["zero_return_days"] = zero_return_days

        # Flag suspicious columns
        report["suspicious_columns"] = missing_pct[missing_pct > 5].index.tolist()

        # Overall pass/fail
        report["passed"] = len(report["suspicious_columns"]) == 0

        # Print summary to console
        self._print_validation_report(report)
        return report

    def _print_validation_report(self, report: dict) -> None:
        """Pretty-print the validation report to the logger."""
        logger.info("=" * 60)
        logger.info("DATA VALIDATION REPORT")
        logger.info("=" * 60)
        logger.info(f"  Shape:            {report['shape'][0]:,} rows × {report['shape'][1]} columns")
        logger.info(f"  Date range:       {report['date_range'][0]} → {report['date_range'][1]}")
        logger.info(f"  Trading days:     {report['trading_days']:,}")
        logger.info(f"  Years of history: {report['years_of_history']}")
        logger.info(f"  Status:           {'✅ PASSED' if report['passed'] else '⚠️  WARNINGS'}")

        if report["suspicious_columns"]:
            logger.warning(
                f"  Columns with >5% missing: {report['suspicious_columns']}"
            )

        if report["zero_return_days"]:
            logger.info(
                f"  Columns with many zero-return days: {report['zero_return_days']}"
            )
        logger.info("=" * 60)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Convenience helpers
    # ─────────────────────────────────────────────────────────────────────────
    def get_returns(self, prices: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute log returns from price data.
        log_return = ln(P_t / P_{t-1})
        If prices=None, fetches ETF data automatically.
        """
        if prices is None:
            prices = self.fetch_etf_data()
        log_returns = np.log(prices / prices.shift(1)).dropna()
        logger.info(f"Log returns computed: {log_returns.shape}")
        return log_returns

    def clear_cache(self) -> None:
        """Delete all cached parquet files to force a fresh download."""
        for p in [self.etf_cache_path, self.macro_cache_path, self.combined_path]:
            if os.path.exists(p):
                os.remove(p)
                logger.info(f"Cache cleared: {p}")
        logger.info("All cache files cleared. Next call will re-fetch from APIs.")

    def cache_info(self) -> dict:
        """Return info about existing cache files."""
        info = {}
        for name, path in {
            "etf":      self.etf_cache_path,
            "macro":    self.macro_cache_path,
            "combined": self.combined_path,
        }.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                mtime   = datetime.fromtimestamp(os.path.getmtime(path))
                info[name] = {
                    "exists":    True,
                    "path":      path,
                    "size_mb":   round(size_mb, 3),
                    "last_modified": str(mtime),
                }
            else:
                info[name] = {"exists": False, "path": path}
        return info


# ─────────────────────────────────────────────────────────────────────────────
# Quick-run entry point: python -m src.data_loader
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    load_dotenv()

    print("\n" + "=" * 60)
    print("  Market Regime Detector — Phase 1: Data Pipeline")
    print("=" * 60 + "\n")

    loader = DataLoader()

    # Show cache status
    print("Cache status:", json.dumps(loader.cache_info(), indent=2))

    # Fetch ETF data
    print("\n[1/3] Fetching ETF prices...")
    etf_data = loader.fetch_etf_data()
    print(etf_data.tail(3))

    # Fetch macro data
    print("\n[2/3] Fetching macro indicators...")
    macro_data = loader.fetch_macro_data()
    if not macro_data.empty:
        print(macro_data.tail(3))

    # Combined dataset
    print("\n[3/3] Building combined dataset...")
    combined = loader.get_combined_data()
    print(combined.tail(3))

    # Validate
    print("\n[VALIDATION]")
    report = loader.validate_data(combined)
    print(f"\nValidation passed: {report['passed']}")
    print(f"Years of history:  {report['years_of_history']}")
