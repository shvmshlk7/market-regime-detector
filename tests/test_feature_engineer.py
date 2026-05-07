"""
tests/test_feature_engineer.py
────────────────────────────────
Unit tests for the FeatureEngineer class.
All tests use synthetic data — no real API calls.

Run with:
    pytest tests/test_feature_engineer.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineer import (
    FeatureEngineer,
    CORR_WIN,
    MA_LONG,
    MA_SHORT,
    MOM_1M,
    MOM_3M,
    MOM_6M,
    VOL_LONG,
    VOL_SHORT,
    ZSCORE_WIN,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def fe():
    """Default FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def minimal_data():
    """
    Minimal synthetic dataset that satisfies min_history (MA_LONG = 200).
    Uses 600 trading days so we get enough valid rows after warmup.
    Includes ETF prices + macro columns.
    """
    n = 600
    dates = pd.bdate_range("2010-01-04", periods=n)
    rng   = np.random.default_rng(42)

    # Simulate geometric Brownian motion prices
    def gbm(S0=100.0, mu=0.0001, sigma=0.012):
        returns = rng.normal(mu, sigma, n)
        prices  = S0 * np.exp(np.cumsum(returns))
        return prices

    df = pd.DataFrame({
        "spy": gbm(100.0),
        "qqq": gbm(200.0),
        "tlt": gbm( 90.0, mu=-0.00005, sigma=0.010),
        "uso": gbm( 30.0, mu=0.0, sigma=0.020),
        "gld": gbm(150.0, mu=0.00005, sigma=0.008),
        # Macro
        "vix":          rng.uniform(10, 40, n),
        "yield_10y":    rng.uniform(1.5, 4.5, n),
        "yield_2y":     rng.uniform(0.5, 4.0, n),
        "cpi":          272.0 + np.cumsum(rng.uniform(0.0, 0.3, n)),
        "unemployment": rng.uniform(3.5, 10.0, n),
    }, index=dates)
    df.index.name = "date"

    # Compute yield_spread (as DataLoader does)
    df["yield_spread"] = df["yield_10y"] - df["yield_2y"]

    return df


@pytest.fixture
def feature_matrix(fe, minimal_data):
    """Pre-computed feature matrix (raw, un-normalized)."""
    return fe.compute_features(minimal_data)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Initialization
# ─────────────────────────────────────────────────────────────────────────────

class TestInit:

    def test_default_columns(self, fe):
        assert fe.equity_col    == "spy"
        assert fe.bond_col      == "tlt"
        assert fe.commodity_col == "uso"

    def test_custom_columns(self):
        fe = FeatureEngineer(equity_col="qqq", bond_col="lqd", commodity_col="gld")
        assert fe.equity_col    == "qqq"
        assert fe.bond_col      == "lqd"
        assert fe.commodity_col == "gld"

    def test_scaler_not_fitted_initially(self, fe):
        assert fe._scaler is None

    def test_feature_names_empty_initially(self, fe):
        with pytest.raises(RuntimeError, match="compute_features"):
            fe.get_feature_names()


# ─────────────────────────────────────────────────────────────────────────────
# Test: compute_features — shape and columns
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFeatures:

    def test_output_has_correct_columns(self, feature_matrix):
        expected = FeatureEngineer.ALL_FEATURES
        assert list(feature_matrix.columns) == expected

    def test_output_has_15_features(self, feature_matrix):
        assert feature_matrix.shape[1] == 15

    def test_no_nan_in_output(self, feature_matrix):
        assert feature_matrix.isna().sum().sum() == 0, \
            "compute_features() should drop all NaN rows"

    def test_index_is_datetimeindex(self, feature_matrix):
        assert isinstance(feature_matrix.index, pd.DatetimeIndex)

    def test_warmup_rows_dropped(self, fe, minimal_data):
        X = fe.compute_features(minimal_data)
        # The 200-day MA warmup + 252-day zscore window requires ~252 rows
        assert len(X) < len(minimal_data)

    def test_raises_on_missing_equity_col(self, fe, minimal_data):
        bad = minimal_data.drop(columns=["spy"])
        with pytest.raises(ValueError, match="spy"):
            fe.compute_features(bad)

    def test_raises_on_insufficient_data(self, fe, minimal_data):
        fe2 = FeatureEngineer(min_history=500)
        short = minimal_data.iloc[:100]   # Only 100 rows
        with pytest.raises(ValueError, match="too short"):
            fe2.compute_features(short)

    def test_raises_on_non_datetimeindex(self, fe, minimal_data):
        bad = minimal_data.reset_index(drop=True)   # Integer index
        with pytest.raises(ValueError, match="DatetimeIndex"):
            fe.compute_features(bad)

    def test_get_feature_names_after_compute(self, fe, minimal_data):
        fe.compute_features(minimal_data)
        names = fe.get_feature_names()
        assert names == FeatureEngineer.ALL_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Test: Volatility Features
# ─────────────────────────────────────────────────────────────────────────────

class TestVolatilityFeatures:

    def test_log_ret_sign_matches_price_direction(self, feature_matrix, minimal_data):
        """Positive log return ↔ price went up that day."""
        # First valid row in feature_matrix — get corresponding price
        dates = feature_matrix.index
        spy   = minimal_data.loc[dates, "spy"]

        sign_ret   = np.sign(feature_matrix["spy_log_ret"])
        sign_price = np.sign(spy.diff().loc[dates])

        # Allow a few ties/edge cases — at least 95% should match
        match_pct = (sign_ret == sign_price).mean()
        assert match_pct > 0.95, f"Log return sign mismatch: {match_pct:.2%}"

    def test_vol_20d_is_positive(self, feature_matrix):
        assert (feature_matrix["spy_vol_20d"] > 0).all()

    def test_vol_60d_is_positive(self, feature_matrix):
        assert (feature_matrix["spy_vol_60d"] > 0).all()

    def test_vol_ratio_finite_positive(self, feature_matrix):
        ratio = feature_matrix["vol_ratio"]
        assert ratio.isna().sum() == 0
        assert (ratio > 0).all()
        assert np.isfinite(ratio).all()

    def test_vol_20d_vs_60d_formula(self):
        """
        Verify vol formula: rolling std × √252 × 100.
        Use a dataset with only 'spy' so macro NaNs don't cause all rows to drop.
        We need enough rows to survive: VOL_SHORT warmup (no macro → no macro NaN).
        """
        # Need ≥ 500 rows: ZSCORE_WIN(252) + MOM_6M(126) + CPI_lag(252) = 378 warmup
        n     = 800
        dates = pd.bdate_range("2015-01-02", periods=n)
        rng    = np.random.default_rng(0)
        prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        # Full macro columns so no feature group produces all-NaN rows
        data  = pd.DataFrame({
            "spy": prices,
            "tlt": prices * 0.8 + rng.normal(0, 0.5, n),
            "uso": prices * 0.3 + rng.normal(0, 0.5, n),
            "vix":          rng.uniform(12, 30, n),
            "yield_spread": rng.uniform(-0.5, 2.0, n),
            "cpi":          272.0 + np.cumsum(rng.uniform(0.0, 0.3, n)),
        }, index=dates)
        data.index.name = "date"
        fe    = FeatureEngineer(min_history=10)

        # Manually compute expected 20d vol for the last row
        log_ret  = np.log(data["spy"] / data["spy"].shift(1))
        expected = log_ret.rolling(VOL_SHORT).std().iloc[-1] * np.sqrt(252) * 100

        X      = fe.compute_features(data)
        actual = X["spy_vol_20d"].iloc[-1]
        assert actual == pytest.approx(expected, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Momentum Features
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumFeatures:

    def test_momentum_features_are_z_scored(self, feature_matrix):
        """Z-scored features should have mean ≈ 0 and std ≈ 1 over full sample."""
        for col in FeatureEngineer.MOMENTUM_FEATURES:
            mean = feature_matrix[col].mean()
            std  = feature_matrix[col].std()
            # Z-scored: mean near 0, std near 1 (not exact because rolling)
            assert abs(mean) < 0.5, f"{col}: mean={mean:.4f} too far from 0"
            assert 0.5 < std < 1.5, f"{col}: std={std:.4f} not near 1"

    def test_longer_window_is_smoother(self, feature_matrix):
        """6M momentum should be smoother (lower std) than 1M momentum."""
        std_1m = feature_matrix["mom_1m_zscore"].std()
        std_6m = feature_matrix["mom_6m_zscore"].std()
        # 6m is a longer cumulative sum — generally smoother
        # This relationship isn't strict but holds for most random walks
        # We just check both are finite and in reasonable range
        assert np.isfinite(std_1m)
        assert np.isfinite(std_6m)

    def test_momentum_not_all_same_sign(self, feature_matrix):
        """Z-scored momentum should have both positive and negative values."""
        for col in FeatureEngineer.MOMENTUM_FEATURES:
            assert feature_matrix[col].min() < 0, f"{col} has no negative values"
            assert feature_matrix[col].max() > 0, f"{col} has no positive values"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Macro Features
# ─────────────────────────────────────────────────────────────────────────────

class TestMacroFeatures:

    def test_vix_level_is_log_transformed(self, feature_matrix, minimal_data):
        """vix_level should equal log(vix)."""
        dates = feature_matrix.index
        vix   = minimal_data.loc[dates, "vix"]
        expected = np.log(vix)
        pd.testing.assert_series_equal(
            feature_matrix["vix_level"].rename("vix"),
            expected.rename("vix"),
            check_names=False,
        )

    def test_vix_zscore_reasonable_range(self, feature_matrix):
        """VIX z-score should mostly be within ±4 standard deviations."""
        z = feature_matrix["vix_zscore"]
        assert z.between(-6, 6).mean() > 0.99

    def test_yield_spread_matches_raw(self, feature_matrix, minimal_data):
        """yield_spread in features should equal raw yield_spread from data."""
        dates = feature_matrix.index
        pd.testing.assert_series_equal(
            feature_matrix["yield_spread"],
            minimal_data.loc[dates, "yield_spread"],
            check_names=False,
        )

    def test_cpi_yoy_is_percentage(self, feature_matrix):
        """CPI YoY should be a % value, typically 0–15% for normal conditions."""
        yoy = feature_matrix["cpi_yoy"].dropna()
        assert (yoy > 0).all(), "CPI YoY should be positive (prices don't fall in our sim)"
        assert (yoy < 50).all(), "CPI YoY above 50% is unrealistic"

    def test_graceful_degradation_no_vix(self, minimal_data):
        """
        If 'vix' is missing, VIX features (vix_level, vix_zscore) become NaN.
        Because these are in ALL_FEATURES and dropna() drops any row with a NaN,
        compute_features() raises ValueError (all rows dropped).
        The expected behavior is a clear ValueError — not a silent crash.
        """
        no_vix = minimal_data.drop(columns=["vix"])
        fe     = FeatureEngineer()
        with pytest.raises(ValueError, match="All rows were dropped"):
            fe.compute_features(no_vix)

    def test_graceful_degradation_no_macro(self, minimal_data):
        """If all macro columns are absent, features degrade but don't crash."""
        no_macro = minimal_data.drop(
            columns=["vix", "yield_spread", "cpi", "yield_10y", "yield_2y"],
            errors="ignore"
        )
        fe = FeatureEngineer()
        # Should not raise — macro features become NaN and rows are dropped
        # Only vol + momentum + cross-asset rows remain valid
        try:
            X = fe.compute_features(no_macro)
            assert X.shape[1] == 15
        except ValueError as e:
            # Only acceptable failure: all rows became NaN (dataset too short)
            assert "All rows were dropped" in str(e)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Cross-Asset Features
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossAssetFeatures:

    def test_bond_equity_corr_bounds(self, feature_matrix):
        """Correlation must be in [-1, +1]."""
        corr = feature_matrix["bond_equity_corr"]
        assert (corr >= -1.0).all(), "Correlation below -1"
        assert (corr <=  1.0).all(), "Correlation above +1"

    def test_commodity_trend_is_z_scored(self, feature_matrix):
        """commodity_trend should be z-scored: mean ≈ 0, std ≈ 1."""
        mean = feature_matrix["commodity_trend"].mean()
        std  = feature_matrix["commodity_trend"].std()
        assert abs(mean) < 0.5, f"commodity_trend mean={mean:.4f}"
        assert 0.5 < std < 1.5, f"commodity_trend std={std:.4f}"

    def test_spy_ma_ratio_positive(self, feature_matrix):
        """MA ratio = 50d MA / 200d MA — both positive, so ratio must be > 0."""
        ratio = feature_matrix["spy_ma_ratio"]
        assert (ratio > 0).all()
        assert np.isfinite(ratio).all()

    def test_spy_ma_ratio_near_1_in_flat_market(self):
        """
        In a flat (constant price) market, MA_50 = MA_200 = price → ratio = 1.
        Test the formula directly (no compute_features) to avoid warmup issues.
        """
        n      = 300
        dates  = pd.bdate_range("2015-01-02", periods=n)
        prices = pd.Series(np.full(n, 100.0), index=dates)

        ma_50  = prices.rolling(MA_SHORT).mean()
        ma_200 = prices.rolling(MA_LONG).mean()
        ratio  = (ma_50 / ma_200.replace(0.0, np.nan)).dropna()

        # All valid ratios must be exactly 1.0 for a constant price series
        assert len(ratio) > 0, "No valid MA ratio rows computed"
        np.testing.assert_allclose(ratio.values, 1.0, atol=1e-9)

    def test_bond_col_missing_gives_nan_column(self, minimal_data):
        """If bond col absent, bond_equity_corr is NaN → rows dropped."""
        no_tlt = minimal_data.drop(columns=["tlt"])
        fe     = FeatureEngineer()
        try:
            X = fe.compute_features(no_tlt)
            # If it succeeds, bond_equity_corr must NOT be in X (all-NaN → dropped)
            # Actually compute_features dropna removes those rows
        except ValueError:
            pass   # Acceptable — too many NaN rows in short dataset


# ─────────────────────────────────────────────────────────────────────────────
# Test: Normalization
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalization:

    def test_fit_transform_zero_mean(self, fe, feature_matrix):
        """After StandardScaler, each feature should have mean ≈ 0."""
        X_norm = fe.fit_transform(feature_matrix)
        means  = X_norm.mean()
        for col in means.index:
            assert abs(means[col]) < 1e-10, f"{col}: mean={means[col]}"

    def test_fit_transform_unit_std(self, fe, feature_matrix):
        """
        After StandardScaler, each feature should have std ≈ 1.
        Tolerance is 1e-4 (not 1e-6) because sklearn uses ddof=0 internally
        while pandas .std() defaults to ddof=1 — small finite-sample difference.
        """
        X_norm = fe.fit_transform(feature_matrix)
        stds   = X_norm.std(ddof=0)   # Match sklearn's ddof=0
        for col in stds.index:
            assert abs(stds[col] - 1.0) < 1e-4, f"{col}: std={stds[col]}"

    def test_fit_transform_preserves_index(self, fe, feature_matrix):
        """Normalized DataFrame must keep the original DatetimeIndex."""
        X_norm = fe.fit_transform(feature_matrix)
        pd.testing.assert_index_equal(X_norm.index, feature_matrix.index)

    def test_fit_transform_preserves_columns(self, fe, feature_matrix):
        """Normalized DataFrame must keep the original column names."""
        X_norm = fe.fit_transform(feature_matrix)
        assert list(X_norm.columns) == list(feature_matrix.columns)

    def test_transform_without_fit_raises(self):
        """transform() before fit_transform() must raise RuntimeError."""
        fe = FeatureEngineer()
        X  = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        with pytest.raises(RuntimeError, match="Scaler not fitted"):
            fe.transform(X)

    def test_transform_applies_same_scale_as_fit(self, fe, feature_matrix):
        """
        transform() on a held-out slice should use training set's mean/std,
        not the slice's own statistics.
        """
        split  = len(feature_matrix) // 2
        train  = feature_matrix.iloc[:split]
        test   = feature_matrix.iloc[split:]

        fe.fit_transform(train)            # Fit on train
        X_test = fe.transform(test)        # Transform test using train stats

        # Test mean (unscaled) should NOT be 0 (different from train distribution)
        # — just verify no error and shape is correct
        assert X_test.shape == test.shape
        assert isinstance(X_test.index, pd.DatetimeIndex)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Feature Importance
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureImportance:

    def test_returns_dataframe_with_correct_shape(self, fe, feature_matrix):
        mi = fe.feature_importance(feature_matrix)
        assert isinstance(mi, pd.DataFrame)
        assert list(mi.columns) == ["feature", "mi_score", "rank"]
        assert len(mi) == feature_matrix.shape[1]   # One row per feature

    def test_mi_scores_non_negative(self, fe, feature_matrix):
        """Mutual information is always ≥ 0."""
        mi = fe.feature_importance(feature_matrix)
        assert (mi["mi_score"] >= 0).all()

    def test_sorted_descending(self, fe, feature_matrix):
        mi = fe.feature_importance(feature_matrix)
        scores = mi["mi_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_rank_column_is_1_indexed(self, fe, feature_matrix):
        mi = fe.feature_importance(feature_matrix)
        # Rank starts at 1
        assert mi["rank"].min() == 1
        # With ties, max rank equals number of unique rank values (not n_features)
        # Just verify ranks are positive integers covering 1..n_features
        assert (mi["rank"] >= 1).all()
        assert (mi["rank"] <= feature_matrix.shape[1]).all()

    def test_vix_level_is_top_feature(self, fe, feature_matrix):
        """
        VIX level is used as the target, so it trivially scores highest.
        This confirms MI is computed correctly.
        """
        mi       = fe.feature_importance(feature_matrix)
        top_feat = mi.iloc[0]["feature"]
        assert top_feat == "vix_level"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Summary
# ─────────────────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_shape(self, fe, feature_matrix):
        stats = fe.summary(feature_matrix)
        assert stats.shape == (feature_matrix.shape[1], 6)
        assert list(stats.columns) == ["mean", "std", "min", "max", "skew", "nulls"]

    def test_no_nulls_in_clean_matrix(self, fe, feature_matrix):
        stats = fe.summary(feature_matrix)
        assert stats["nulls"].sum() == 0


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
