"""
tests/test_portfolio_optimizer.py
─────────────────────────────────
Unit tests for the PortfolioOptimizer class.
"""

import pytest
import pandas as pd
import numpy as np

from src.portfolio_optimizer import PortfolioOptimizer


@pytest.fixture
def sample_prices():
    """Generates a synthetic price dataframe for 4 assets over 300 days."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=300)
    
    # 4 assets with different characteristics
    # SPY: steady growth
    # TLT: slight negative drift, negatively correlated
    # GLD: independent
    # QQQ: high growth, high vol
    
    returns = np.random.normal(0, 0.01, size=(300, 4))
    returns[:, 0] += 0.0005 # SPY positive drift
    returns[:, 1] -= 0.0002 # TLT negative drift
    returns[:, 3] += 0.0010 # QQQ high drift
    
    # Make TLT negatively correlated with SPY/QQQ
    returns[:, 1] -= returns[:, 0] * 0.5 
    
    prices = pd.DataFrame(np.exp(np.cumsum(returns, axis=0)), index=dates, columns=["SPY", "TLT", "GLD", "QQQ"])
    prices.iloc[0] = 100.0  # normalize start
    return prices


@pytest.fixture
def optimizer():
    # Wider bounds for testing (allow up to 60%) to ensure it can optimize
    # Note: 4 assets, min 10%, max 60% is feasible
    return PortfolioOptimizer(min_weight=0.10, max_weight=0.60, risk_free_rate=0.01, lookback_days=50)


class TestPortfolioOptimizer:

    def test_init_defaults(self):
        po = PortfolioOptimizer()
        assert po.min_weight == 0.05
        assert po.max_weight == 0.30
        assert po.risk_free_rate == 0.02
        assert po.lookback_days == 252

    def test_optimize_empty_prices_raises(self, optimizer):
        with pytest.raises(ValueError, match="Price dataframe cannot be empty"):
            optimizer.optimize(pd.DataFrame(), "Bull")
            
    def test_optimize_infeasible_bounds(self):
        # 4 assets, min 0.30 means total is 1.20 > 1.0 (infeasible)
        po = PortfolioOptimizer(min_weight=0.40, max_weight=0.50)
        prices = pd.DataFrame(np.random.rand(100, 4), columns=["A", "B", "C", "D"])
        
        with pytest.raises(ValueError, match="Infeasible: min_weight"):
            po.optimize(prices, "Bull")
            
    def test_optimize_bull_regime(self, optimizer, sample_prices):
        weights = optimizer.optimize(sample_prices.iloc[-100:], "Bull")
        
        # Check output structure
        assert isinstance(weights, dict)
        assert set(weights.keys()) == set(sample_prices.columns)
        
        # Check sum to 1
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
        
        # Check bounds
        for w in weights.values():
            assert w >= optimizer.min_weight - 1e-4
            assert w <= optimizer.max_weight + 1e-4

    def test_optimize_bear_regime(self, optimizer, sample_prices):
        # Bear regime should trigger minimum volatility
        weights = optimizer.optimize(sample_prices.iloc[-100:], "Bear")
        
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
        for w in weights.values():
            assert w >= optimizer.min_weight - 1e-4
            assert w <= optimizer.max_weight + 1e-4

    def test_optimize_fallback_on_error(self, optimizer):
        # Data that will break covariance or EMA calculation (e.g., all identical rows or too short)
        bad_prices = pd.DataFrame(np.ones((2, 4)), columns=["A", "B", "C", "D"])
        weights = optimizer.optimize(bad_prices, "Bull")
        
        # Should fallback to equal weights (1/4 = 0.25)
        for w in weights.values():
            assert pytest.approx(w) == 0.25

    def test_get_historical_weights(self, optimizer, sample_prices):
        dates = sample_prices.index
        # Create regimes: first 100 days Bull, then 100 Bear, then 100 Sideways
        regimes = pd.Series("Bull", index=dates)
        regimes.iloc[100:200] = "Bear"
        regimes.iloc[200:] = "Sideways"
        
        hist_weights = optimizer.get_historical_weights(sample_prices, regimes)
        
        # Check shape
        assert hist_weights.shape == sample_prices.shape
        assert (hist_weights.index == sample_prices.index).all()
        
        # Check no NaNs
        assert not hist_weights.isna().any().any()
        
        # Check sum to 1 row-wise
        row_sums = hist_weights.sum(axis=1)
        assert np.allclose(row_sums, 1.0)
        
        # We know lookback_days = 50. Before index 50, it should be equal weight
        eq_weight = 0.25
        assert pytest.approx(hist_weights.iloc[0]["SPY"]) == eq_weight
        assert pytest.approx(hist_weights.iloc[49]["SPY"]) == eq_weight
        
        # The weights should be flat (unchanged) when regime doesn't change
        # E.g., index 150 to 190 (Bear regime)
        diffs_bear = hist_weights.iloc[150:190].diff().dropna()
        assert (diffs_bear == 0).all().all()
        
        # There should be changes precisely when regime switches 
        # (at index 100 and 200, though since we test equality, 
        # let's just make sure it changes somewhere)
        diffs_total = hist_weights.diff().dropna()
        assert not (diffs_total == 0).all().all()

    def test_get_historical_weights_misaligned_index_raises(self, optimizer, sample_prices):
        regimes = pd.Series("Bull", index=pd.bdate_range("2030-01-01", periods=10))
        with pytest.raises(ValueError, match="common date index"):
            optimizer.get_historical_weights(sample_prices, regimes)
