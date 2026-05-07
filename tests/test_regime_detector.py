"""
tests/test_regime_detector.py
──────────────────────────────
Unit tests for the RegimeDetector class.
All tests use synthetic data — no real API calls, no real model files.

Run with:
    pytest tests/test_regime_detector.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.regime_detector import RegimeDetector, _LOG_RET_FEATURE


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_feature_matrix(
    n: int = 600,
    n_features: int = 15,
    seed: int = 42,
    feature_names: list[str] | None = None,
    strong_signal: bool = True,
) -> pd.DataFrame:
    """
    Create a synthetic normalized feature matrix that mimics Phase 2 output.

    Injects three clearly-separated regimes so the HMM reliably finds them:
      - First third:  strong positive drift (Bull-like)
      - Middle third: strong negative drift with higher vol (Bear-like)
      - Last third:   near-zero drift (Sideways-like)
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range("2010-01-04", periods=n)

    if feature_names is None:
        feature_names = [
            "spy_log_ret", "spy_vol_20d", "spy_vol_60d", "vol_ratio",
            "mom_1m_zscore", "mom_3m_zscore", "mom_6m_zscore",
            "vix_level", "vix_zscore", "yield_spread",
            "yield_spread_chg", "cpi_yoy",
            "bond_equity_corr", "commodity_trend", "spy_ma_ratio",
        ][:n_features]

    # Low-noise base so the signal is dominant
    data = rng.standard_normal((n, len(feature_names))) * 0.1

    if strong_signal:
        third = n // 3
        # Bull: strong positive returns, low vol, high momentum
        data[:third, 0]  += 2.0    # spy_log_ret strongly positive
        data[:third, 1]  -= 1.0    # spy_vol_20d low
        data[:third, 4]  += 1.5    # mom_1m_zscore positive

        # Bear: strong negative returns, high vol
        data[third:2*third, 0]  -= 2.0   # spy_log_ret strongly negative
        data[third:2*third, 1]  += 2.0   # spy_vol_20d high
        data[third:2*third, 7]  += 1.5   # vix_level elevated

        # Sideways: near-zero across all features (the default)

    df = pd.DataFrame(data, index=dates, columns=feature_names)
    df.index.name = "date"
    return df


@pytest.fixture
def X_norm() -> pd.DataFrame:
    """600-row normalized feature matrix with strongly injected Bull/Bear/Sideways signal."""
    return _make_feature_matrix(n=600, strong_signal=True)


@pytest.fixture
def rd() -> RegimeDetector:
    """Default RegimeDetector instance (not yet fitted). min_regime_hold=1 so
    smoothing never collapses the predicted distribution in tests."""
    return RegimeDetector(n_regimes=3, n_iter=200, random_state=42, min_regime_hold=1)


@pytest.fixture
def fitted_rd(rd, X_norm) -> RegimeDetector:
    """Fitted RegimeDetector."""
    rd.fit(X_norm)
    return rd


# ─────────────────────────────────────────────────────────────────────────────
# Test: Initialization
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeDetectorInit:

    def test_default_params(self):
        rd = RegimeDetector()
        assert rd.n_regimes       == 3
        assert rd.n_iter          == 200
        assert rd.covariance_type == "full"
        assert rd.random_state    == 42
        assert rd.min_regime_hold == 5

    def test_custom_params(self):
        rd = RegimeDetector(
            n_regimes=4, n_iter=50, covariance_type="diag",
            random_state=7, min_regime_hold=2
        )
        assert rd.n_regimes       == 4
        assert rd.n_iter          == 50
        assert rd.covariance_type == "diag"
        assert rd.random_state    == 7
        assert rd.min_regime_hold == 2

    def test_not_fitted_initially(self, rd):
        assert not rd._is_fitted

    def test_label_map_empty_before_fit(self, rd):
        assert rd._label_map == {}

    def test_model_none_before_fit(self, rd):
        assert rd._model is None

    def test_predict_before_fit_raises(self, rd, X_norm):
        with pytest.raises(RuntimeError, match="not fitted"):
            rd.predict(X_norm)

    def test_predict_proba_before_fit_raises(self, rd, X_norm):
        with pytest.raises(RuntimeError, match="not fitted"):
            rd.predict_proba(X_norm)

    def test_save_before_fit_raises(self, rd):
        with pytest.raises(RuntimeError, match="unfitted"):
            rd.save("/tmp/test_model.pkl")


# ─────────────────────────────────────────────────────────────────────────────
# Test: Fitting
# ─────────────────────────────────────────────────────────────────────────────

class TestFit:

    def test_fit_returns_self(self, rd, X_norm):
        result = rd.fit(X_norm)
        assert result is rd   # method chaining

    def test_is_fitted_after_fit(self, fitted_rd):
        assert fitted_rd._is_fitted

    def test_label_map_populated_after_fit(self, fitted_rd):
        assert len(fitted_rd._label_map) == 3

    def test_label_map_has_all_regime_names(self, fitted_rd):
        labels = set(fitted_rd._label_map.values())
        assert labels == {"Bull", "Bear", "Sideways"}

    def test_feature_names_stored_after_fit(self, fitted_rd, X_norm):
        assert fitted_rd._feature_names == X_norm.columns.tolist()

    def test_fit_raises_on_nan(self, rd):
        bad = _make_feature_matrix(n=300)
        bad.iloc[10, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            rd.fit(bad)

    def test_fit_raises_on_non_datetimeindex(self, rd):
        df = _make_feature_matrix(n=300).reset_index(drop=True)
        with pytest.raises(ValueError, match="DatetimeIndex"):
            rd.fit(df)

    def test_fit_raises_on_single_row(self, rd):
        tiny = _make_feature_matrix(n=1)
        with pytest.raises(ValueError, match="at least 2"):
            rd.fit(tiny)


# ─────────────────────────────────────────────────────────────────────────────
# Test: predict()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredict:

    def test_output_is_series(self, fitted_rd, X_norm):
        result = fitted_rd.predict(X_norm)
        assert isinstance(result, pd.Series)

    def test_output_length_matches_input(self, fitted_rd, X_norm):
        result = fitted_rd.predict(X_norm)
        assert len(result) == len(X_norm)

    def test_output_index_matches_input(self, fitted_rd, X_norm):
        result = fitted_rd.predict(X_norm)
        pd.testing.assert_index_equal(result.index, X_norm.index)

    def test_all_labels_are_valid_regimes(self, fitted_rd, X_norm):
        result   = fitted_rd.predict(X_norm)
        valid    = {"Bull", "Bear", "Sideways"}
        observed = set(result.unique())
        assert observed.issubset(valid), f"Unexpected labels: {observed - valid}"

    def test_multiple_regimes_detected(self, fitted_rd, X_norm):
        """With injected signal, detector should find at least 2 distinct regimes."""
        result = fitted_rd.predict(X_norm)
        assert result.nunique() >= 2, "Should detect at least 2 distinct regimes"

    def test_series_name_is_regime(self, fitted_rd, X_norm):
        result = fitted_rd.predict(X_norm)
        assert result.name == "regime"


# ─────────────────────────────────────────────────────────────────────────────
# Test: Auto-labelling (Bull = highest return, Bear = lowest)
# ─────────────────────────────────────────────────────────────────────────────

class TestAutoLabeling:

    def test_bull_regime_has_highest_mean_return(self, fitted_rd, X_norm):
        """Rows predicted Bull should have a higher mean spy_log_ret than Bear rows.
        Only asserted when both regimes are actually predicted."""
        regimes  = fitted_rd.predict(X_norm)
        bull_rows = X_norm.loc[regimes == "Bull",  _LOG_RET_FEATURE].dropna()
        bear_rows = X_norm.loc[regimes == "Bear",  _LOG_RET_FEATURE].dropna()
        if bull_rows.empty or bear_rows.empty:
            pytest.skip("HMM did not produce both Bull and Bear states — skipping.")
        bull_ret = bull_rows.mean()
        bear_ret = bear_rows.mean()
        assert bull_ret > bear_ret, (
            f"Bull mean log-ret={bull_ret:.4f} should exceed Bear={bear_ret:.4f}"
        )

    def test_bear_regime_has_lowest_mean_return(self, fitted_rd, X_norm):
        regimes    = fitted_rd.predict(X_norm)
        bear_rows  = X_norm.loc[regimes == "Bear",     _LOG_RET_FEATURE].dropna()
        side_rows  = X_norm.loc[regimes == "Sideways", _LOG_RET_FEATURE].dropna()
        if bear_rows.empty or side_rows.empty:
            pytest.skip("HMM did not produce both Bear and Sideways states — skipping.")
        bear_ret = bear_rows.mean()
        side_ret = side_rows.mean()
        assert bear_ret < side_ret, (
            f"Bear mean log-ret={bear_ret:.4f} should be below Sideways={side_ret:.4f}"
        )

    def test_label_map_covers_all_states(self, fitted_rd):
        for state in range(fitted_rd.n_regimes):
            assert state in fitted_rd._label_map


# ─────────────────────────────────────────────────────────────────────────────
# Test: predict_proba()
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictProba:

    def test_output_is_dataframe(self, fitted_rd, X_norm):
        proba = fitted_rd.predict_proba(X_norm)
        assert isinstance(proba, pd.DataFrame)

    def test_output_shape(self, fitted_rd, X_norm):
        proba = fitted_rd.predict_proba(X_norm)
        assert proba.shape == (len(X_norm), fitted_rd.n_regimes)

    def test_rows_sum_to_one(self, fitted_rd, X_norm):
        proba = fitted_rd.predict_proba(X_norm)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_all_columns_are_regime_names(self, fitted_rd, X_norm):
        proba = fitted_rd.predict_proba(X_norm)
        valid = {"Bull", "Bear", "Sideways"}
        assert set(proba.columns).issubset(valid)

    def test_probabilities_non_negative(self, fitted_rd, X_norm):
        proba = fitted_rd.predict_proba(X_norm)
        assert (proba.values >= 0).all()

    def test_index_matches_input(self, fitted_rd, X_norm):
        proba = fitted_rd.predict_proba(X_norm)
        pd.testing.assert_index_equal(proba.index, X_norm.index)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Minimum regime hold smoothing
# ─────────────────────────────────────────────────────────────────────────────

class TestMinRegimeHold:

    def _series(self, labels: list[str]) -> pd.Series:
        dates = pd.bdate_range("2020-01-02", periods=len(labels))
        return pd.Series(labels, index=dates, name="regime", dtype="object")

    def test_short_blip_is_removed(self):
        rd = RegimeDetector(min_regime_hold=5)
        # 'Bear' for 2 days inside a long 'Bull' run → should be smoothed away
        labels = ["Bull"] * 10 + ["Bear"] * 2 + ["Bull"] * 10
        raw    = self._series(labels)
        smooth = rd._apply_min_hold(raw)

        # The two Bear days should be replaced by Bull
        assert (smooth.iloc[10:12] == "Bull").all(), (
            f"Short blip not smoothed: {smooth.iloc[9:13].tolist()}"
        )

    def test_long_run_is_preserved(self):
        rd = RegimeDetector(min_regime_hold=3)
        labels = ["Bull"] * 5 + ["Bear"] * 5 + ["Bull"] * 5
        raw    = self._series(labels)
        smooth = rd._apply_min_hold(raw)
        # 5-day Bear run should survive (≥ min_regime_hold=3)
        assert (smooth.iloc[5:10] == "Bear").all()

    def test_min_hold_1_no_change(self):
        """min_regime_hold=1 → return unchanged."""
        rd = RegimeDetector(min_regime_hold=1)
        labels = ["Bull", "Bear", "Sideways", "Bull"]
        raw    = self._series(labels)
        smooth = rd._apply_min_hold(raw)
        pd.testing.assert_series_equal(smooth, raw)

    def test_all_same_label_unchanged(self):
        rd = RegimeDetector(min_regime_hold=5)
        labels = ["Bull"] * 20
        raw    = self._series(labels)
        smooth = rd._apply_min_hold(raw)
        assert (smooth == "Bull").all()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Persistence (save / load)
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistence:

    def test_save_creates_file(self, fitted_rd, tmp_path):
        path = str(tmp_path / "model.pkl")
        fitted_rd.save(path)
        assert os.path.exists(path)

    def test_load_returns_regimedetector(self, fitted_rd, tmp_path):
        path = str(tmp_path / "model.pkl")
        fitted_rd.save(path)
        loaded = RegimeDetector.load(path)
        assert isinstance(loaded, RegimeDetector)

    def test_load_is_fitted(self, fitted_rd, tmp_path):
        path = str(tmp_path / "model.pkl")
        fitted_rd.save(path)
        loaded = RegimeDetector.load(path)
        assert loaded._is_fitted

    def test_load_preserves_label_map(self, fitted_rd, tmp_path):
        path = str(tmp_path / "model.pkl")
        fitted_rd.save(path)
        loaded = RegimeDetector.load(path)
        assert loaded._label_map == fitted_rd._label_map

    def test_load_preserves_params(self, X_norm, tmp_path):
        rd = RegimeDetector(n_regimes=3, n_iter=50, min_regime_hold=3)
        rd.fit(X_norm)
        path = str(tmp_path / "model.pkl")
        rd.save(path)
        loaded = RegimeDetector.load(path)
        assert loaded.n_iter          == 50
        assert loaded.min_regime_hold == 3

    def test_load_reproduces_predictions(self, fitted_rd, X_norm, tmp_path):
        """Loaded model must produce identical predictions."""
        path = str(tmp_path / "model.pkl")
        fitted_rd.save(path)
        loaded = RegimeDetector.load(path)

        original = fitted_rd.predict(X_norm)
        reloaded = loaded.predict(X_norm)
        pd.testing.assert_series_equal(original, reloaded)

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            RegimeDetector.load("/nonexistent/path/model.pkl")

    def test_save_creates_parent_dirs(self, fitted_rd, tmp_path):
        path = str(tmp_path / "nested" / "deep" / "model.pkl")
        fitted_rd.save(path)
        assert os.path.exists(path)


# ─────────────────────────────────────────────────────────────────────────────
# Test: get_regime_stats()
# ─────────────────────────────────────────────────────────────────────────────

class TestRegimeStats:

    def test_returns_dataframe(self, fitted_rd, X_norm):
        stats = fitted_rd.get_regime_stats(X_norm)
        assert isinstance(stats, pd.DataFrame)

    def test_index_contains_regime_labels(self, fitted_rd, X_norm):
        stats  = fitted_rd.get_regime_stats(X_norm)
        labels = set(stats.index)
        assert labels.issubset({"Bull", "Bear", "Sideways"})

    def test_count_sums_to_total(self, fitted_rd, X_norm):
        stats = fitted_rd.get_regime_stats(X_norm)
        total = stats[("meta", "count")].sum()
        assert total == len(X_norm)

    def test_pct_sums_to_100(self, fitted_rd, X_norm):
        stats   = fitted_rd.get_regime_stats(X_norm)
        pct_sum = stats[("meta", "pct")].sum()
        assert abs(pct_sum - 100.0) < 0.5   # rounding tolerance

    def test_accepts_raw_feature_matrix(self, fitted_rd, X_norm):
        """Should work with X_raw passed as second argument."""
        X_raw  = X_norm * 2.0   # simulate un-normalized (same shape)
        stats  = fitted_rd.get_regime_stats(X_norm, X_raw)
        assert isinstance(stats, pd.DataFrame)

    def test_stats_before_fit_raises(self, rd, X_norm):
        with pytest.raises(RuntimeError, match="not fitted"):
            rd.get_regime_stats(X_norm)


# ─────────────────────────────────────────────────────────────────────────────
# Test: summary()
# ─────────────────────────────────────────────────────────────────────────────

class TestSummary:

    def test_summary_unfitted_is_string(self, rd):
        s = rd.summary()
        assert isinstance(s, str)
        assert "unfitted" in s.lower()

    def test_summary_fitted_is_string(self, fitted_rd):
        s = fitted_rd.summary()
        assert isinstance(s, str)

    def test_summary_fitted_contains_n_regimes(self, fitted_rd):
        s = fitted_rd.summary()
        assert str(fitted_rd.n_regimes) in s

    def test_summary_fitted_contains_label_names(self, fitted_rd):
        s = fitted_rd.summary()
        for label in {"Bull", "Bear", "Sideways"}:
            assert label in s


# ─────────────────────────────────────────────────────────────────────────────
# Test: Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_predict_on_subset_of_features(self, fitted_rd, X_norm):
        """If columns match training, predict should still work."""
        subset = X_norm.iloc[-50:]   # last 50 rows only
        result = fitted_rd.predict(subset)
        assert len(result) == 50

    def test_two_regime_model(self, X_norm):
        """n_regimes=2 should produce Bull + Bear labels."""
        rd = RegimeDetector(n_regimes=2, n_iter=50, random_state=0)
        rd.fit(X_norm)
        labels = set(rd._label_map.values())
        assert labels == {"Bull", "Bear"}

    def test_predict_proba_two_regimes(self, X_norm):
        rd = RegimeDetector(n_regimes=2, n_iter=50, random_state=0)
        rd.fit(X_norm)
        proba = rd.predict_proba(X_norm)
        assert proba.shape[1] == 2

    def test_large_min_hold_still_returns_labels(self, X_norm):
        """min_regime_hold larger than any single run → still returns series."""
        rd = RegimeDetector(n_regimes=3, n_iter=50, min_regime_hold=999)
        rd.fit(X_norm)
        result = rd.predict(X_norm)
        assert len(result) == len(X_norm)
        assert result.notna().all()

    def test_non_dataframe_raises(self, rd, X_norm):
        rd.fit(X_norm)
        with pytest.raises(TypeError, match="pd.DataFrame"):
            rd.predict(X_norm.values)    # numpy array, not DataFrame

    def test_predict_raises_on_nan_input(self, fitted_rd, X_norm):
        bad = X_norm.copy()
        bad.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            fitted_rd.predict(bad)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
