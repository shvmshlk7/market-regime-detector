"""
src/feature_engineer.py
───────────────────────
FeatureEngineer class — Phase 2 of the Market Regime Detector.

Transforms raw ETF + macro data (from DataLoader) into a normalized
feature matrix ready for GaussianHMM training in Phase 3.

Feature Groups (15 features total):
┌─────────────────┬──────────────────────────────────────────────────────────┐
│ Group           │ Features                                                 │
├─────────────────┼──────────────────────────────────────────────────────────┤
│ Volatility (4)  │ spy_log_ret, spy_vol_20d, spy_vol_60d, vol_ratio         │
│ Momentum (3)    │ mom_1m_zscore, mom_3m_zscore, mom_6m_zscore              │
│ Macro (5)       │ vix_level, vix_zscore, yield_spread,                     │
│                 │ yield_spread_chg, cpi_yoy                                │
│ Cross-asset (3) │ bond_equity_corr, commodity_trend, spy_ma_ratio          │
└─────────────────┴──────────────────────────────────────────────────────────┘

Usage:
    from src.data_loader import DataLoader
    from src.feature_engineer import FeatureEngineer

    loader = DataLoader()
    data   = loader.get_combined_data()       # ETF prices + macro

    fe     = FeatureEngineer()
    X_raw  = fe.compute_features(data)        # Raw feature matrix (15 cols)
    X_norm = fe.fit_transform(X_raw)          # StandardScaler normalized
    report = fe.feature_importance(X_raw)     # Mutual information scores
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Rolling window constants (in trading days)
# ─────────────────────────────────────────────────────────────────────────────
VOL_SHORT   = 20     # Short rolling volatility window
VOL_LONG    = 60     # Long rolling volatility window
MOM_1M      = 21     # ~1 month of trading days
MOM_3M      = 63     # ~3 months of trading days
MOM_6M      = 126    # ~6 months of trading days
ZSCORE_WIN  = 252    # 1-year rolling window for z-score normalization
CORR_WIN    = 60     # Bond-equity correlation window
MA_SHORT    = 50     # Short moving average
MA_LONG     = 200    # Long moving average (sets the min_history floor)
SPREAD_CHG  = 20     # Yield spread change look-back


class FeatureEngineer:
    """
    Builds a normalized 15-feature matrix for HMM regime detection.

    Parameters
    ----------
    equity_col : str
        Primary equity price column name. Default: 'spy'.
    bond_col : str
        Long-duration bond column name (for correlation). Default: 'tlt'.
    commodity_col : str
        Commodity/oil ETF column name. Default: 'uso'.
    min_history : int
        Minimum rows required before features are valid. Rows before
        this threshold are NaN and will be dropped by compute_features().
        Default: 200 (dictated by the 200-day MA requirement).
    """

    # ── Feature name groups — useful for subsetting the matrix ───────────────
    VOLATILITY_FEATURES  = ["spy_log_ret", "spy_vol_20d", "spy_vol_60d", "vol_ratio"]
    MOMENTUM_FEATURES    = ["mom_1m_zscore", "mom_3m_zscore", "mom_6m_zscore"]
    MACRO_FEATURES       = ["vix_level", "vix_zscore", "yield_spread",
                            "yield_spread_chg", "cpi_yoy"]
    CROSS_ASSET_FEATURES = ["bond_equity_corr", "commodity_trend", "spy_ma_ratio"]
    ALL_FEATURES         = (VOLATILITY_FEATURES + MOMENTUM_FEATURES
                            + MACRO_FEATURES + CROSS_ASSET_FEATURES)

    def __init__(
        self,
        equity_col:    str = "spy",
        bond_col:      str = "tlt",
        commodity_col: str = "uso",
        min_history:   int = MA_LONG,
    ):
        self.equity_col    = equity_col
        self.bond_col      = bond_col
        self.commodity_col = commodity_col
        self.min_history   = min_history

        self._scaler:       Optional[StandardScaler] = None
        self._feature_names: list[str] = []

        logger.info(
            f"FeatureEngineer initialized | "
            f"equity={equity_col} | bond={bond_col} | commodity={commodity_col} | "
            f"min_history={min_history}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the full feature matrix from the combined ETF + macro dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Output of DataLoader.get_combined_data().
            Required columns: lowercase equity column (e.g. 'spy').
            Optional columns: 'tlt', 'uso', 'vix', 'yield_spread', 'cpi'.

        Returns
        -------
        pd.DataFrame
            Feature matrix with DatetimeIndex. NaN warmup rows are dropped.
            Shape: (trading_days - warmup, 15).
        """
        logger.info(f"Computing feature matrix | input shape: {data.shape}")
        self._validate_input(data)

        # Start with an empty frame sharing the same index
        features = pd.DataFrame(index=data.index)

        features = self._add_volatility_features(features, data)
        features = self._add_momentum_features(features, data)
        features = self._add_macro_features(features, data)
        features = self._add_cross_asset_features(features, data)

        # Drop NaN rows from the warmup period
        n_before = len(features)
        features = features.dropna()
        n_dropped = n_before - len(features)

        if len(features) == 0:
            raise ValueError(
                f"All rows were dropped as NaN. "
                f"Your dataset may be too short (need 250+ trading days after the "
                f"{n_dropped}-row warmup period). "
                f"Try extending START_DATE in config.py."
            )

        self._feature_names = features.columns.tolist()
        logger.info(
            f"Feature matrix ready | shape: {features.shape} | "
            f"warmup dropped: {n_dropped} rows | "
            f"date range: {features.index[0].date()} → {features.index[-1].date()}"
        )
        return features

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit a StandardScaler on X and return the normalized matrix.

        Preserves the DatetimeIndex and column names.
        Call this on training data. For new/test data, use transform().

        Returns
        -------
        pd.DataFrame — same shape as X, values z-scored per feature.
        """
        self._scaler = StandardScaler()
        scaled = self._scaler.fit_transform(X.values)
        result = pd.DataFrame(scaled, index=X.index, columns=X.columns)
        logger.info(f"StandardScaler fitted + applied | shape: {result.shape}")
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a previously fitted scaler to new data (no re-fitting).

        Prevents lookahead bias: the scaler's mean/std come only from
        training data. Must call fit_transform() before this.
        """
        if self._scaler is None:
            raise RuntimeError(
                "Scaler not fitted. Call fit_transform() on training data first."
            )
        scaled = self._scaler.transform(X.values)
        return pd.DataFrame(scaled, index=X.index, columns=X.columns)

    def feature_importance(
        self,
        X: pd.DataFrame,
        n_neighbors: int = 5,
    ) -> pd.DataFrame:
        """
        Estimate feature relevance via mutual information (MI) regression.

        Uses VIX level (log-transformed) as the target — a proxy for
        market stress. Features with high MI score tend to reliably track
        bear/bull regimes. This is for exploratory analysis only.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix from compute_features().
        n_neighbors : int
            k-NN parameter for MI estimation. Higher = smoother estimate.

        Returns
        -------
        pd.DataFrame
            Columns: [feature, mi_score, rank]
            Sorted by mi_score descending.
        """
        # Pick the best available stress proxy as the target
        if "vix_level" in X.columns:
            target     = X["vix_level"].values
            target_lbl = "vix_level (log VIX)"
        elif "spy_vol_20d" in X.columns:
            target     = X["spy_vol_20d"].values
            target_lbl = "spy_vol_20d (stress proxy)"
        else:
            target     = X.iloc[:, 0].values
            target_lbl = X.columns[0]

        # Drop rows with ANY NaN before passing to sklearn
        valid_mask = ~np.isnan(X.values).any(axis=1) & ~np.isnan(target)
        X_clean    = X.values[valid_mask]
        y_clean    = target[valid_mask]

        if len(X_clean) < 50:
            raise ValueError(
                f"Too few valid rows ({len(X_clean)}) for MI estimation. "
                "Check for excessive NaNs in the feature matrix."
            )

        mi_scores = mutual_info_regression(
            X_clean, y_clean, n_neighbors=n_neighbors, random_state=42
        )

        result = pd.DataFrame({
            "feature":  X.columns.tolist(),
            "mi_score": mi_scores.round(4),
        })
        result["rank"] = result["mi_score"].rank(ascending=False).astype(int)
        result = result.sort_values("mi_score", ascending=False).reset_index(drop=True)

        logger.info(f"Mutual information vs '{target_lbl}':")
        logger.info(f"\n{result.to_string(index=False)}")
        return result

    def summary(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame of per-feature descriptive statistics.
        Useful for sanity-checking the feature matrix before HMM training.

        Returns
        -------
        pd.DataFrame with columns: mean, std, min, max, skew, nulls
        """
        stats = pd.DataFrame({
            "mean":  X.mean().round(4),
            "std":   X.std().round(4),
            "min":   X.min().round(4),
            "max":   X.max().round(4),
            "skew":  X.skew().round(4),
            "nulls": X.isna().sum(),
        })
        logger.info(f"\nFeature summary:\n{stats.to_string()}")
        return stats

    def get_feature_names(self) -> list[str]:
        """Return feature names from the last compute_features() call."""
        if not self._feature_names:
            raise RuntimeError(
                "No features computed yet. Call compute_features() first."
            )
        return self._feature_names.copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Group 1: Volatility Features
    # ─────────────────────────────────────────────────────────────────────────

    def _add_volatility_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute volatility-based features.

        spy_log_ret  — Daily log return: ln(P_t / P_{t-1})
                       The HMM uses this to distinguish Bull (positive drift)
                       from Bear (negative drift) regimes.

        spy_vol_20d  — Rolling 20-day realized volatility, annualized (%).
                       Short-term vol spike is the primary bear signal.

        spy_vol_60d  — Rolling 60-day realized volatility, annualized (%).
                       Background / structural vol level.

        vol_ratio    — spy_vol_20d / spy_vol_60d
                       > 1.0 → recent vol elevated vs background → stress
                       < 1.0 → calm recent vol vs background → calm regime
                       This ratio cleanly separates regime transitions.
        """
        prices  = data[self.equity_col]
        log_ret = np.log(prices / prices.shift(1))

        features["spy_log_ret"] = log_ret
        features["spy_vol_20d"] = log_ret.rolling(VOL_SHORT).std() * np.sqrt(252) * 100
        features["spy_vol_60d"] = log_ret.rolling(VOL_LONG).std()  * np.sqrt(252) * 100

        # Protect against zero denominator
        vol_60d_safe = features["spy_vol_60d"].replace(0.0, np.nan)
        features["vol_ratio"] = features["spy_vol_20d"] / vol_60d_safe

        logger.debug(
            f"Volatility features | non-null rows: "
            f"{features['vol_ratio'].notna().sum()}"
        )
        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Group 2: Momentum Features
    # ─────────────────────────────────────────────────────────────────────────

    def _add_momentum_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute momentum z-scores.

        Cumulative log returns over 1M / 3M / 6M windows, z-scored
        using a trailing 252-day (1 year) rolling mean and std. This
        expresses momentum as "how unusual is this return relative to
        the past year?" — removing the absolute scale.

        Positive z-score → above-average momentum → bull regime
        Negative z-score → below-average momentum → bear/corrective regime

        The rolling z-score window prevents lookahead: at date t, only
        data up to t is used. Safe for walk-forward training.
        """
        prices  = data[self.equity_col]
        log_ret = np.log(prices / prices.shift(1))

        windows = {
            "1m": MOM_1M,   # 21 trading days
            "3m": MOM_3M,   # 63 trading days
            "6m": MOM_6M,   # 126 trading days
        }

        for label, window in windows.items():
            cum_ret   = log_ret.rolling(window).sum()   # Cumulative log return
            roll_mean = cum_ret.rolling(ZSCORE_WIN).mean()
            roll_std  = cum_ret.rolling(ZSCORE_WIN).std()

            # Avoid division by near-zero std (very rare but can happen in flat markets)
            roll_std_safe = roll_std.replace(0.0, np.nan)
            features[f"mom_{label}_zscore"] = (cum_ret - roll_mean) / roll_std_safe

        logger.debug("Momentum features computed.")
        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Group 3: Macro Features
    # ─────────────────────────────────────────────────────────────────────────

    def _add_macro_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute macro environment features.

        Gracefully degrades: if FRED data is absent (no API key), macro
        columns are set to NaN, then dropped by compute_features()'s
        dropna(). The HMM will still train on volatility + momentum.

        vix_level        — log(VIX). Log-transform stabilizes the right
                           skew (VIX spikes to 80 in crashes but usually
                           hovers around 15–25).

        vix_zscore       — (VIX - rolling_mean) / rolling_std over 252d.
                           Captures whether fear is elevated vs its own
                           recent history.

        yield_spread     — 10Y - 2Y Treasury spread (already in DataLoader).
                           Negative (inverted) = historically precedes recession.

        yield_spread_chg — 20-day change in yield spread.
                           Captures the direction of curve movement, which
                           often leads equity regime changes.

        cpi_yoy          — CPI year-over-year % change.
                           High and rising inflation constrains Fed policy,
                           historically correlates with bear equity regimes.
        """
        # ── VIX ─────────────────────────────────────────────────────────────
        if "vix" in data.columns:
            vix = data["vix"].clip(lower=1e-6)               # Guard log(0)
            features["vix_level"] = np.log(vix)              # Log-transform

            roll_mean = vix.rolling(ZSCORE_WIN).mean()
            roll_std  = vix.rolling(ZSCORE_WIN).std().replace(0.0, np.nan)
            features["vix_zscore"] = (vix - roll_mean) / roll_std
            logger.debug("VIX features computed.")
        else:
            logger.warning(
                "'vix' column not found in data — VIX features will be NaN. "
                "Set FRED_API_KEY in .env to enable macro features."
            )
            features["vix_level"]  = np.nan
            features["vix_zscore"] = np.nan

        # ── Yield Spread ─────────────────────────────────────────────────────
        if "yield_spread" in data.columns:
            features["yield_spread"]     = data["yield_spread"]
            features["yield_spread_chg"] = data["yield_spread"].diff(SPREAD_CHG)
            logger.debug("Yield spread features computed.")
        else:
            logger.warning("'yield_spread' column not found — yield features will be NaN.")
            features["yield_spread"]     = np.nan
            features["yield_spread_chg"] = np.nan

        # ── CPI YoY ───────────────────────────────────────────────────────────
        if "cpi" in data.columns:
            cpi = data["cpi"]
            # 252-trading-day lag approximates 1 calendar year for daily data
            cpi_lag = cpi.shift(252).replace(0.0, np.nan)
            features["cpi_yoy"] = (cpi / cpi_lag - 1.0) * 100.0
            logger.debug("CPI YoY feature computed.")
        else:
            logger.warning("'cpi' column not found — cpi_yoy feature will be NaN.")
            features["cpi_yoy"] = np.nan

        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Group 4: Cross-Asset Features
    # ─────────────────────────────────────────────────────────────────────────

    def _add_cross_asset_features(
        self, features: pd.DataFrame, data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute cross-asset relationship features.

        bond_equity_corr  — Rolling 60d Pearson correlation between SPY and
                            TLT log returns.
                            Negative → flight to safety → bear signal.
                            Positive → risk-on, bonds and equities rising →
                            unusual but seen in early bull markets.

        commodity_trend   — USO (oil) 20-day cumulative return, z-scored vs
                            trailing 252-day history.
                            Strong oil trend → economic expansion → bull.
                            Collapsing oil → demand destruction → bear/recession.

        spy_ma_ratio      — SPY 50-day MA / SPY 200-day MA.
                            Golden cross (> 1.0) → trend following buy signal.
                            Death cross (< 1.0) → long-term trend breakdown.
                            This is one of the most widely used trend indicators
                            by institutional traders.
        """
        eq_ret = np.log(data[self.equity_col] / data[self.equity_col].shift(1))

        # ── Bond-Equity Correlation ──────────────────────────────────────────
        if self.bond_col in data.columns:
            bond_ret = np.log(data[self.bond_col] / data[self.bond_col].shift(1))
            features["bond_equity_corr"] = eq_ret.rolling(CORR_WIN).corr(bond_ret)
            logger.debug("Bond-equity correlation computed.")
        else:
            logger.warning(
                f"'{self.bond_col}' column not found — "
                "bond_equity_corr will be NaN."
            )
            features["bond_equity_corr"] = np.nan

        # ── Commodity Trend ─────────────────────────────────────────────────
        if self.commodity_col in data.columns:
            comm_ret  = np.log(data[self.commodity_col] / data[self.commodity_col].shift(1))
            cum_ret   = comm_ret.rolling(VOL_SHORT).sum()    # 20-day cumulative
            roll_mean = cum_ret.rolling(ZSCORE_WIN).mean()
            roll_std  = cum_ret.rolling(ZSCORE_WIN).std().replace(0.0, np.nan)
            features["commodity_trend"] = (cum_ret - roll_mean) / roll_std
            logger.debug("Commodity trend feature computed.")
        else:
            logger.warning(
                f"'{self.commodity_col}' column not found — "
                "commodity_trend will be NaN."
            )
            features["commodity_trend"] = np.nan

        # ── SPY MA Ratio ─────────────────────────────────────────────────────
        prices = data[self.equity_col]
        ma_50  = prices.rolling(MA_SHORT).mean()
        ma_200 = prices.rolling(MA_LONG).mean().replace(0.0, np.nan)
        features["spy_ma_ratio"] = ma_50 / ma_200
        logger.debug("SPY MA ratio computed.")

        return features

    # ─────────────────────────────────────────────────────────────────────────
    # Validation
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_input(self, data: pd.DataFrame) -> None:
        """
        Check input DataFrame before computing features.
        Raises ValueError with clear message on first failure found.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError(
                "data.index must be a DatetimeIndex. "
                "Use DataLoader.get_combined_data() which sets this correctly."
            )

        if self.equity_col not in data.columns:
            raise ValueError(
                f"Required equity column '{self.equity_col}' not found. "
                f"Available columns: {data.columns.tolist()}. "
                f"Pass equity_col='<your_col>' to FeatureEngineer()."
            )

        if len(data) < self.min_history:
            raise ValueError(
                f"Dataset too short: {len(data)} rows, need ≥ {self.min_history}. "
                f"Extend the date range in config.py (currently START_DATE = "
                f"'{data.index[0].date()}')."
            )

        if data[self.equity_col].isna().all():
            raise ValueError(
                f"Equity column '{self.equity_col}' is entirely NaN. "
                "Check that ETF data was fetched correctly via DataLoader."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Quick-run entry point: python -m src.feature_engineer
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()

    from src.data_loader import DataLoader

    print("\n" + "=" * 60)
    print("  Market Regime Detector — Phase 2: Feature Engineering")
    print("=" * 60 + "\n")

    loader = DataLoader()
    data   = loader.get_combined_data()

    fe    = FeatureEngineer()
    X_raw = fe.compute_features(data)

    print("\n[FEATURE MATRIX]")
    print(X_raw.tail(5).to_string())

    print("\n[SUMMARY STATISTICS]")
    stats = fe.summary(X_raw)
    print(stats.to_string())

    print("\n[NORMALIZED MATRIX — first 3 rows]")
    X_norm = fe.fit_transform(X_raw)
    print(X_norm.head(3).to_string())

    print("\n[FEATURE IMPORTANCE — Mutual Information]")
    mi = fe.feature_importance(X_raw)
    print(mi.to_string(index=False))

    print(f"\n✅ Phase 2 complete. Feature matrix: {X_raw.shape}")
    print("─" * 60 + "\n")
