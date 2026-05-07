"""
portfolio_optimizer.py
──────────────────────
PortfolioOptimizer class for Phase 4.

Uses PyPortfolioOpt to generate optimal asset allocations based on the
market regime. Connects the dots between the HMM output and actual
investment decisions.
"""

import logging
from typing import Dict

import pandas as pd
import numpy as np

# PyPortfolioOpt imports
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

# Import strict bounds from config
from src.config import MIN_SINGLE_WEIGHT, MAX_SINGLE_WEIGHT, ETF_TICKERS

logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Given a dataframe of historical prices and a target market regime,
    calculates optimal portfolio weights using Modern Portfolio Theory.
    """

    def __init__(
        self,
        min_weight: float = MIN_SINGLE_WEIGHT,
        max_weight: float = MAX_SINGLE_WEIGHT,
        risk_free_rate: float = 0.02,
        lookback_days: int = 252,
    ):
        """
        Initialize the optimizer.
        
        Parameters
        ----------
        min_weight : float
            Minimum allocation to any single asset (e.g., 0.05 for 5%).
        max_weight : float
            Maximum allocation to any single asset (e.g., 0.30 for 30%).
        risk_free_rate : float
            Annualized risk-free rate used for Sharpe ratio calculation.
        lookback_days : int
            Number of trading days to look back for calculating mu and S. 
            Default is 252 (approx 1 year).
        """
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days
        
        logger.debug(f"PortfolioOptimizer initialized with bounds [{min_weight}, {max_weight}]")

    def optimize(self, prices: pd.DataFrame, regime: str) -> Dict[str, float]:
        """
        Calculate optimal weights for the given price history and target regime.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data. Should optionally only include the recent 
            `lookback_days` of data for the most relevant estimates.
        regime : str
            The target regime ("Bull", "Bear", "Sideways").
            
        Returns
        -------
        dict
            Dictionary mapping ticker symbols to their optimal weight (0.0 to 1.0).
        """
        if prices is None or prices.empty:
            raise ValueError("Price dataframe cannot be empty")
            
        # Ensure we don't have unrealistic bounds given number of assets
        n_assets = len(prices.columns)
        if self.min_weight * n_assets > 1.0:
            raise ValueError(f"Infeasible: min_weight {self.min_weight} * {n_assets} assets > 100%")
        if self.max_weight * n_assets < 1.0:
            raise ValueError(f"Infeasible: max_weight {self.max_weight} * {n_assets} assets < 100%")

        # 1. Calculate Expected Returns (mu) and Covariance (S)
        # We use exponential moving average for expected returns to put more weight on recent data
        try:
            mu = expected_returns.ema_historical_return(prices, span=self.lookback_days)
            # Ledoit-Wolf shrinkage gives a robust covariance matrix, stable even with fewer samples
            S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        except Exception as e:
            logger.error(f"Failed to calculate mu or S: {e}")
            return self._fallback_weights(prices.columns)

        # 2. Instantiate Efficient Frontier with constraints
        # Ensure solvers act quietly
        ef = EfficientFrontier(mu, S, weight_bounds=(self.min_weight, self.max_weight))
        
        try:
            # 3. Regime-conditional optimization objective
            if regime == "Bull":
                # Maximize return given a target risk, or simply Maximize Sharpe.
                # Maximize Sharpe is standard and balances growth with risk.
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
            
            elif regime == "Bear":
                # Minimize volatility strictly to preserve capital
                ef.min_volatility()
                
            elif regime == "Sideways":
                # Maximize Sharpe ratio to find best risk-reward
                # We could alternatively use efficient_risk to target a specific volatility
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                
            else:
                logger.warning(f"Unknown regime '{regime}', defaulting to max_sharpe")
                ef.max_sharpe(risk_free_rate=self.risk_free_rate)
                
            # 4. Clean and return weights
            weights = ef.clean_weights()
            return dict(weights)
            
        except Exception as e:
            logger.warning(f"Optimization failed for regime '{regime}' ({e}). Falling back to equal weight.")
            return self._fallback_weights(prices.columns)

    def _fallback_weights(self, columns) -> Dict[str, float]:
        """Returns equal weights if optimization fails."""
        n = len(columns)
        return {col: 1.0 / n for col in columns}

    def get_historical_weights(self, prices: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        """
        Generate a time-series of target portfolio weights.
        Only re-optimizes when the detected regime changes to limit turnover.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Full history of ETF prices.
        regimes : pd.Series
            Time series of regime labels ("Bull", "Bear", "Sideways").
            
        Returns
        -------
        pd.DataFrame
            DataFrame of portfolio weights spanning the same index as the inputs.
        """
        logger.info("Calculating historical portfolio weights...")
        
        # Align indexes to be safe (inner join to keep days both have data)
        common_idx = prices.index.intersection(regimes.index)
        if len(common_idx) == 0:
            raise ValueError("Prices and regimes must share a common date index")
            
        prices = prices.loc[common_idx]
        regimes = regimes.loc[common_idx]
        
        # Prepare output dataframe
        weights_df = pd.DataFrame(index=common_idx, columns=prices.columns, dtype=float)
        
        current_regime = None
        current_weights = self._fallback_weights(prices.columns)
        
        # Start optimization only after we have enough lookback data
        start_idx = min(self.lookback_days, len(common_idx)-1)
        
        for i, dt in enumerate(common_idx):
            if i < start_idx:
                # Pre-warmup period: equal weight
                weights_df.loc[dt] = current_weights
                continue
                
            day_regime = regimes.iloc[i]
            
            # Rebalance if regime changes
            if day_regime != current_regime:
                # Get the trailing window of prices
                window_prices = prices.iloc[i - self.lookback_days + 1 : i + 1]
                
                logger.debug(f"[{dt.date()}] Regime changed {current_regime} -> {day_regime}. Re-optimizing.")
                
                # Optimize
                opt_w = self.optimize(window_prices, day_regime)
                current_weights = opt_w
                current_regime = day_regime
            
            # Record weight for the day
            weights_df.loc[dt] = current_weights
            
        logger.info(f"Historical weight calculation complete. Shape: {weights_df.shape}")
        return weights_df
