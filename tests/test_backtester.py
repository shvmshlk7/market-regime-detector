"""
tests/test_backtester.py
────────────────────────
Unit tests for the Backtester class.
"""

import pytest
import pandas as pd
import numpy as np

from src.backtester import Backtester

@pytest.fixture
def prices():
    """Simple 2-asset price dataframe."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=100)
    
    # Create simple upward trending prices
    df = pd.DataFrame(index=dates)
    df["A"] = 100 * np.exp(np.linspace(0, 0.5, 100))
    df["B"] = 100 * np.exp(np.linspace(0, 0.2, 100))
    return df

@pytest.fixture
def weights(prices):
    """Simple switching weights."""
    dates = prices.index
    df = pd.DataFrame(index=dates)
    
    # 100% Asset A for first half, 100% Asset B for second half
    df["A"] = np.concatenate([np.ones(50), np.zeros(50)])
    df["B"] = np.concatenate([np.zeros(50), np.ones(50)])
    return df

class TestBacktester:

    def test_init_sets_transaction_cost(self):
        bt = Backtester(transaction_cost=0.01)
        assert bt.transaction_cost == 0.01
        
    def test_run_backtest_produces_portfolio(self, prices, weights):
        bt = Backtester(transaction_cost=0.0)
        pf = bt.run_backtest(prices, weights)
        
        assert pf is not None
        # In a frictionless world holding upward assets, final value > initial 100
        assert pf.value().iloc[-1] > 100.0

    def test_transaction_costs_reduce_returns(self, prices, weights):
        bt_zero = Backtester(transaction_cost=0.0)
        pf_zero = bt_zero.run_backtest(prices, weights)
        
        bt_high = Backtester(transaction_cost=0.05)
        pf_high = bt_high.run_backtest(prices, weights)
        
        # Total return with friction should be lower than frictionless
        assert pf_high.total_return() < pf_zero.total_return()

    def test_run_benchmark(self, prices):
        bt = Backtester(transaction_cost=0.0)
        # Using "A" as benchmark
        pf_bench = bt.run_benchmark(prices, benchmark_tickers=["A"])
        
        assert pf_bench is not None
        # Benchmark holding purely A should match return of A
        A_return = (prices["A"].iloc[-1] / prices["A"].iloc[0]) - 1.0
        assert pytest.approx(pf_bench.total_return(), rel=1e-3) == A_return

    def test_get_metrics(self, prices, weights):
        bt = Backtester(transaction_cost=0.0)
        bt.run_backtest(prices, weights)
        bt.run_benchmark(prices, benchmark_tickers=["A"])
        
        metrics = bt.get_metrics()
        
        assert isinstance(metrics, pd.DataFrame)
        assert "Strategy" in metrics.columns
        assert "Benchmark" in metrics.columns
        assert "Total Return [%]" in metrics.index

    def test_get_metrics_fails_without_run(self):
        bt = Backtester()
        with pytest.raises(ValueError, match="Must run_backtest"):
            bt.get_metrics()

    def test_mismatched_indexes_handled_safely(self, prices):
        bt = Backtester()
        # Create weights with only 50 rows
        dates = prices.index[:50]
        w = pd.DataFrame(1.0, index=dates, columns=prices.columns)
        
        pf = bt.run_backtest(prices, w)
        # The portfolio Should align properties and run for 50 days 
        assert len(pf.value()) == 50
