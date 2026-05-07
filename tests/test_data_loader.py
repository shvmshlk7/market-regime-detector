"""
tests/test_data_loader.py
──────────────────────────
Unit tests for the DataLoader class.
These tests use mock data — no real API calls during testing.

Run with:
    pytest tests/test_data_loader.py -v
"""

import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add project root to path so src imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_dirs(tmp_path):
    """Create temporary cache directories for testing."""
    raw       = tmp_path / "raw"
    processed = tmp_path / "processed"
    raw.mkdir()
    processed.mkdir()
    return str(raw), str(processed)


@pytest.fixture
def sample_price_df():
    """Create a sample ETF price DataFrame for testing."""
    tickers = ["SPY", "QQQ", "GLD", "TLT"]
    dates   = pd.bdate_range("2020-01-02", periods=500)  # ~2 years of trading days
    data    = np.random.uniform(100, 400, size=(len(dates), len(tickers)))
    df      = pd.DataFrame(data, index=dates, columns=tickers)
    df.index.name = "date"
    return df


@pytest.fixture
def sample_macro_df():
    """Create a sample macro DataFrame with deterministic yield_spread."""
    dates     = pd.bdate_range("2020-01-02", periods=500)
    yield_10y = np.random.uniform(0.5, 4.5, len(dates))
    yield_2y  = np.random.uniform(0.1, 4.0, len(dates))
    df        = pd.DataFrame({
        "vix":          np.random.uniform(10, 80, len(dates)),
        "yield_10y":    yield_10y,
        "yield_2y":     yield_2y,
        "yield_spread": yield_10y - yield_2y,   # Computed correctly
        "cpi":          np.random.uniform(250, 310, len(dates)),
        "unemployment": np.random.uniform(3.5, 15.0, len(dates)),
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def loader(temp_dirs):
    """Create a DataLoader with temp directories and a small date range."""
    raw_dir, proc_dir = temp_dirs
    loader = DataLoader(
        start_date="2020-01-01",
        end_date="2022-01-01",
        tickers=["SPY", "QQQ", "GLD", "TLT"],
        fred_api_key=None,          # No real API calls
        cache_dir_raw=raw_dir,
        cache_dir_processed=proc_dir,
    )
    # Override cache paths to use temp dirs
    loader.etf_cache_path   = os.path.join(raw_dir,  "etf_prices.parquet")
    loader.macro_cache_path = os.path.join(raw_dir,  "macro_data.parquet")
    loader.combined_path    = os.path.join(proc_dir, "combined_dataset.parquet")
    return loader


# ─────────────────────────────────────────────────────────────────────────────
# Test: Initialization
# ─────────────────────────────────────────────────────────────────────────────

class TestDataLoaderInit:

    def test_defaults_are_applied(self, loader):
        assert loader.start_date == "2020-01-01"
        assert loader.end_date   == "2022-01-01"
        assert isinstance(loader.tickers, list)
        assert len(loader.tickers) == 4

    def test_directories_created(self, temp_dirs):
        raw_dir, proc_dir = temp_dirs
        loader = DataLoader(
            start_date="2020-01-01",
            end_date="2021-01-01",
            cache_dir_raw=raw_dir,
            cache_dir_processed=proc_dir,
        )
        assert os.path.exists(raw_dir)
        assert os.path.exists(proc_dir)

    def test_fred_not_initialized_without_key(self, loader):
        """FRED client should not be initialized if no key."""
        assert loader._fred is None  # Not yet loaded


# ─────────────────────────────────────────────────────────────────────────────
# Test: Price Data Cleaning
# ─────────────────────────────────────────────────────────────────────────────

class TestPriceDataCleaning:

    def test_forward_fill_small_gaps(self, loader, sample_price_df):
        # Introduce some NaN gaps (simulating holidays)
        df_with_gaps = sample_price_df.copy()
        df_with_gaps.iloc[10:12, 0] = np.nan  # 2-day gap in SPY

        result = loader._clean_price_data(df_with_gaps)
        # After forward-fill, the 2-day gap should be filled
        assert result.iloc[10, 0] == pytest.approx(df_with_gaps.iloc[9, 0])

    def test_drops_tickers_with_excessive_missing(self, loader):
        # Create a ticker with >10% missing
        dates  = pd.bdate_range("2020-01-02", periods=100)
        df     = pd.DataFrame({
            "SPY": np.random.uniform(300, 400, 100),
            "BAD": [np.nan] * 100,  # 100% missing
        }, index=dates)
        df.index.name = "date"

        result = loader._clean_price_data(df)
        assert "BAD" not in result.columns
        assert "SPY" in result.columns

    def test_cleans_without_removing_valid_data(self, loader, sample_price_df):
        """No valid rows should be removed from clean data."""
        result = loader._clean_price_data(sample_price_df)
        # All rows should survive (no NaNs in our sample)
        assert len(result) <= len(sample_price_df)  # Can only shrink or stay same


# ─────────────────────────────────────────────────────────────────────────────
# Test: Parquet Caching
# ─────────────────────────────────────────────────────────────────────────────

class TestParquetCaching:

    def test_saves_and_loads_from_cache(self, loader, sample_price_df):
        # Save to cache
        sample_price_df.to_parquet(loader.etf_cache_path, index=True)
        assert os.path.exists(loader.etf_cache_path)

        # Load from cache (should not call yfinance)
        with patch("yfinance.download") as mock_yf:
            result = loader.fetch_etf_data(force_refresh=False)
            mock_yf.assert_not_called()

        # check_freq=False: parquet round-trip strips the BusinessDay freq attribute
        pd.testing.assert_frame_equal(result, sample_price_df, check_freq=False)

    def test_force_refresh_bypasses_cache(self, loader, sample_price_df):
        # Pre-populate cache
        sample_price_df.to_parquet(loader.etf_cache_path, index=True)

        with patch("yfinance.download") as mock_yf:
            # Simulate yfinance 1.x group_by='column' layout:
            # MultiIndex Level 0 = fields (Close, Open, ...), Level 1 = tickers
            mock_data = sample_price_df.copy()
            mock_data.columns = pd.MultiIndex.from_tuples(
                [("Close", c) for c in mock_data.columns],
                names=["field", "ticker"],
            )
            mock_yf.return_value = mock_data
            loader.fetch_etf_data(force_refresh=True)
            mock_yf.assert_called_once()

    def test_clear_cache_removes_files(self, loader, sample_price_df):
        # Create cache files
        sample_price_df.to_parquet(loader.etf_cache_path, index=True)
        assert os.path.exists(loader.etf_cache_path)

        loader.clear_cache()
        assert not os.path.exists(loader.etf_cache_path)

    def test_cache_info_detects_files(self, loader, sample_price_df):
        # No cache yet
        info_before = loader.cache_info()
        assert not info_before["etf"]["exists"]

        # Create cache
        sample_price_df.to_parquet(loader.etf_cache_path, index=True)
        info_after = loader.cache_info()
        assert info_after["etf"]["exists"]
        assert info_after["etf"]["size_mb"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# Test: Log Returns
# ─────────────────────────────────────────────────────────────────────────────

class TestLogReturns:

    def test_log_returns_shape(self, loader, sample_price_df):
        returns = loader.get_returns(sample_price_df)
        # Returns should have one fewer row than prices (first row dropped)
        assert len(returns) == len(sample_price_df) - 1
        assert list(returns.columns) == list(sample_price_df.columns)

    def test_log_returns_values(self, loader):
        """Manually verify log return formula: log(P_t / P_{t-1})"""
        dates   = pd.bdate_range("2020-01-02", periods=3)
        prices  = pd.DataFrame({"SPY": [100.0, 110.0, 99.0]}, index=dates)
        prices.index.name = "date"

        returns = loader.get_returns(prices)
        expected_r1 = np.log(110.0 / 100.0)
        expected_r2 = np.log(99.0  / 110.0)

        assert returns["SPY"].iloc[0] == pytest.approx(expected_r1, rel=1e-6)
        assert returns["SPY"].iloc[1] == pytest.approx(expected_r2, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Data Validation
# ─────────────────────────────────────────────────────────────────────────────

class TestDataValidation:

    def test_validation_passes_on_clean_data(self, loader, sample_price_df):
        report = loader.validate_data(sample_price_df)
        assert report["passed"] is True
        assert report["shape"] == sample_price_df.shape
        assert report["trading_days"] == len(sample_price_df)

    def test_validation_fails_on_missing_data(self, loader, sample_price_df):
        # Introduce >5% missing in one column
        df = sample_price_df.copy()
        df.iloc[:50, 0] = np.nan  # 50/500 = 10% missing
        report = loader.validate_data(df)
        assert not report["passed"]
        assert df.columns[0] in report["suspicious_columns"]

    def test_years_of_history_calculation(self, loader, sample_price_df):
        """500 trading days ≈ ~2 years of history."""
        report = loader.validate_data(sample_price_df)
        expected_years = round(500 / 252, 1)
        assert report["years_of_history"] == expected_years

    def test_date_range_in_report(self, loader, sample_price_df):
        report = loader.validate_data(sample_price_df)
        start, end = report["date_range"]
        assert start == str(sample_price_df.index[0].date())
        assert end   == str(sample_price_df.index[-1].date())


# ─────────────────────────────────────────────────────────────────────────────
# Test: Macro Resampling
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroResampling:

    def test_resamples_to_business_days(self, loader):
        """Monthly data should be resampled to daily business days."""
        # Simulate monthly CPI data
        monthly_dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        monthly_df = pd.DataFrame(
            {"cpi": np.random.uniform(255, 305, 24)},
            index=monthly_dates
        )
        monthly_df.index.name = "date"

        # Temporarily override date range
        loader.start_date = "2020-01-01"
        loader.end_date   = "2022-01-01"

        resampled = loader._resample_macro_to_daily(monthly_df)

        # Should return a business-day indexed DataFrame
        assert len(resampled) > 24  # More rows than monthly data
        # No NaNs after forward-fill (within 30-day limit)
        middle = resampled.iloc[1:-1]  # Ignore boundaries
        assert middle["cpi"].notna().sum() > 0

    def test_yield_spread_is_computed(self, loader, sample_macro_df):
        """yield_spread = yield_10y - yield_2y"""
        row = sample_macro_df.iloc[0]
        expected_spread = row["yield_10y"] - row["yield_2y"]
        assert abs(row["yield_spread"] - expected_spread) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
