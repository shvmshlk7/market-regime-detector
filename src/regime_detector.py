"""
src/regime_detector.py
───────────────────────
RegimeDetector class — Phase 3 of the Market Regime Detector.

Trains a Gaussian HMM on the 15-feature normalized matrix produced by
FeatureEngineer (Phase 2) and assigns one of three market regime labels
to every trading day:

    ┌──────────┬─────────────────────────────────────────────────────────┐
    │  Label   │  Characteristics                                        │
    ├──────────┼─────────────────────────────────────────────────────────┤
    │  Bull    │  Positive log-return drift, low volatility, rising MAs  │
    │  Bear    │  Negative drift, high vol, VIX spike, falling momentum  │
    │  Sideways│  Near-zero drift, moderate vol, no clear trend          │
    └──────────┴─────────────────────────────────────────────────────────┘

States come out of the HMM as integers (0, 1, 2). They are
**auto-labelled** by sorting on each state's mean spy_log_ret:
  - Highest mean return  → Bull
  - Lowest  mean return  → Bear
  - Middle               → Sideways

Usage:
    from src.feature_engineer import FeatureEngineer
    from src.regime_detector  import RegimeDetector

    fe     = FeatureEngineer()
    X_raw  = fe.compute_features(data)
    X_norm = fe.fit_transform(X_raw)

    rd = RegimeDetector(n_regimes=3)
    rd.fit(X_norm)

    regimes      = rd.predict(X_norm)          # pd.Series of "Bull"/"Bear"/"Sideways"
    proba        = rd.predict_proba(X_norm)    # pd.DataFrame of regime probabilities
    stats        = rd.get_regime_stats(X_norm, X_raw)
    rd.save("models/regime_hmm.pkl")

    rd2 = RegimeDetector.load("models/regime_hmm.pkl")
"""

import logging
import os
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_N_REGIMES       = 3
_DEFAULT_N_ITER          = 200
_DEFAULT_COV_TYPE        = "full"
_DEFAULT_RANDOM_STATE    = 42
_DEFAULT_MIN_HOLD        = 5          # Days before a regime change is confirmed
_LOG_RET_FEATURE         = "spy_log_ret"   # Feature used for auto-labelling
_REGIME_NAMES: dict[int, str] = {0: "Bull", 1: "Bear", 2: "Sideways"}


class RegimeDetector:
    """
    Gaussian HMM-based market regime classifier.

    Parameters
    ----------
    n_regimes : int
        Number of hidden states. Default: 3 (Bull / Bear / Sideways).
    n_iter : int
        Maximum EM iterations for hmmlearn. Default: 200.
    covariance_type : str
        HMM covariance structure: 'full' | 'diag' | 'tied' | 'spherical'.
        'full' is richest — each state gets its own full covariance matrix.
        Default: 'full'.
    random_state : int
        Seed for reproducibility. Default: 42.
    min_regime_hold : int
        Minimum consecutive trading days before a detected regime is
        confirmed. Short blips (< min_regime_hold) are smoothed out by
        forward-filling the previous label. Default: 5.
    refit_window_days : int
        Placeholder for Phase 5 walk-forward refit window.
        Not used in this phase. Default: 252.
    """

    def __init__(
        self,
        n_regimes:        int = _DEFAULT_N_REGIMES,
        n_iter:           int = _DEFAULT_N_ITER,
        covariance_type:  str = _DEFAULT_COV_TYPE,
        random_state:     int = _DEFAULT_RANDOM_STATE,
        min_regime_hold:  int = _DEFAULT_MIN_HOLD,
        refit_window_days: int = 252,
    ):
        self.n_regimes         = n_regimes
        self.n_iter            = n_iter
        self.covariance_type   = covariance_type
        self.random_state      = random_state
        self.min_regime_hold   = min_regime_hold
        self.refit_window_days = refit_window_days

        self._model:      Optional[hmm.GaussianHMM] = None
        self._label_map:  dict[int, str]            = {}   # hmm_state → regime name
        self._is_fitted:  bool                      = False
        self._feature_names: list[str]              = []

        logger.info(
            f"RegimeDetector initialised | n_regimes={n_regimes} | "
            f"cov={covariance_type} | n_iter={n_iter} | "
            f"min_hold={min_regime_hold}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def fit(self, X_norm: pd.DataFrame) -> "RegimeDetector":
        """
        Train the Gaussian HMM on the normalized feature matrix.

        Parameters
        ----------
        X_norm : pd.DataFrame
            Output of FeatureEngineer.fit_transform() — shape (T, 15),
            DatetimeIndex, no NaN values.

        Returns
        -------
        self — enables method chaining.
        """
        self._validate_input(X_norm, require_fitted=False)

        logger.info(
            f"Fitting GaussianHMM | shape={X_norm.shape} | "
            f"n_regimes={self.n_regimes} | cov={self.covariance_type}"
        )

        self._model = hmm.GaussianHMM(
            n_components      = self.n_regimes,
            covariance_type   = self.covariance_type,
            n_iter            = self.n_iter,
            random_state      = self.random_state,
            verbose           = False,
        )

        self._model.fit(X_norm.values)
        self._feature_names = X_norm.columns.tolist()

        # Auto-label states by mean spy_log_ret (Bull = highest)
        self._label_map = self._auto_label(X_norm)
        self._is_fitted  = True

        logger.info(
            f"HMM fitted | converged={self._model.monitor_.converged} | "
            f"label_map={self._label_map}"
        )
        return self

    def predict(self, X_norm: pd.DataFrame) -> pd.Series:
        """
        Predict regime labels for each row in X_norm.

        Applies minimum-hold smoothing to eliminate very-short blips that
        the raw Viterbi path sometimes produces at transitions.

        Parameters
        ----------
        X_norm : pd.DataFrame
            Normalized feature matrix (same columns as training data).

        Returns
        -------
        pd.Series[str]
            Regime label per trading day ('Bull' / 'Bear' / 'Sideways').
            Same DatetimeIndex as X_norm.
        """
        self._validate_input(X_norm, require_fitted=True)

        raw_states = self._model.predict(X_norm.values)
        raw_labels = pd.Series(
            [self._label_map[s] for s in raw_states],
            index  = X_norm.index,
            name   = "regime",
            dtype  = "object",
        )

        smoothed = self._apply_min_hold(raw_labels)
        logger.info(
            f"Regimes predicted | shape={X_norm.shape[0]} | "
            f"counts={smoothed.value_counts().to_dict()}"
        )
        return smoothed

    def predict_proba(self, X_norm: pd.DataFrame) -> pd.DataFrame:
        """
        Return posterior regime probabilities for each trading day.

        Uses the HMM's forward algorithm (log-probability of each state
        sequence) rather than the Viterbi hard assignment. This gives
        a continuous confidence signal useful for portfolio blending.

        Parameters
        ----------
        X_norm : pd.DataFrame
            Normalized feature matrix.

        Returns
        -------
        pd.DataFrame
            Shape (T, n_regimes). Columns are regime names.
            Each row sums to 1.0.
        """
        self._validate_input(X_norm, require_fitted=True)

        log_proba = self._model.predict_proba(X_norm.values)   # (T, n_regimes)

        # Re-order columns so they follow Bull / Bear / Sideways regardless
        # of the internal HMM state ordering
        inv_map    = {v: k for k, v in self._label_map.items()}
        col_order  = [n for n in ["Bull", "Bear", "Sideways"] if n in inv_map]
        col_order += [n for n in inv_map if n not in col_order]   # any extras

        # Build DataFrame with internal-state column names first
        internal_cols = [self._label_map[i] for i in range(self.n_regimes)]
        df = pd.DataFrame(log_proba, index=X_norm.index, columns=internal_cols)

        # Reorder into canonical order (duplicates handled via reindex)
        df = df.reindex(columns=col_order, fill_value=0.0)
        logger.info(f"Regime probabilities computed | shape={df.shape}")
        return df

    def get_regime_stats(
        self,
        X_norm: pd.DataFrame,
        X_raw:  Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Compute per-regime summary statistics.

        Parameters
        ----------
        X_norm : pd.DataFrame
            Normalized feature matrix (used to assign regime labels).
        X_raw : pd.DataFrame, optional
            Un-normalized feature matrix.  If provided, statistics are
            computed on the raw values (more interpretable).
            Must share the same DatetimeIndex as X_norm.

        Returns
        -------
        pd.DataFrame
            MultiIndex (regime, statistic) × features.
            Statistics: mean, std, count, pct.
        """
        self._validate_input(X_norm, require_fitted=True)

        regimes = self.predict(X_norm)
        source  = X_raw if X_raw is not None else X_norm
        source  = source.loc[regimes.index]   # align on shared index

        rows = []
        total = len(regimes)
        for label in sorted(set(self._label_map.values())):
            mask      = regimes == label
            subset    = source[mask]
            n         = len(subset)
            if n == 0:
                continue
            stat_row = {
                ("meta", "count"): n,
                ("meta", "pct"):   round(n / total * 100, 1),
            }
            for col in source.columns:
                stat_row[(col, "mean")] = round(subset[col].mean(), 4)
                stat_row[(col, "std")]  = round(subset[col].std(),  4)
            rows.append((label, stat_row))

        # Build multi-index DataFrame
        index  = pd.Index([r[0] for r in rows], name="regime")
        data   = [r[1] for r in rows]
        result = pd.DataFrame(data, index=index)
        result.columns = pd.MultiIndex.from_tuples(result.columns)
        logger.info(f"Regime stats computed | shape={result.shape}")
        return result

    def save(self, path: str) -> None:
        """
        Persist the fitted model and label map to disk using joblib.

        Parameters
        ----------
        path : str
            File path (recommended extension: .pkl).
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Cannot save an unfitted model. Call fit() first."
            )
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "model":         self._model,
            "label_map":     self._label_map,
            "feature_names": self._feature_names,
            "params": {
                "n_regimes":         self.n_regimes,
                "n_iter":            self.n_iter,
                "covariance_type":   self.covariance_type,
                "random_state":      self.random_state,
                "min_regime_hold":   self.min_regime_hold,
                "refit_window_days": self.refit_window_days,
            },
        }
        joblib.dump(payload, path)
        logger.info(f"RegimeDetector saved → {path}")

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        """
        Load a previously saved RegimeDetector from disk.

        Parameters
        ----------
        path : str
            Path to the .pkl file created by save().

        Returns
        -------
        RegimeDetector
            Fully fitted instance, ready to predict.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        payload = joblib.load(path)
        params  = payload["params"]

        rd = cls(
            n_regimes         = params["n_regimes"],
            n_iter            = params["n_iter"],
            covariance_type   = params["covariance_type"],
            random_state      = params["random_state"],
            min_regime_hold   = params["min_regime_hold"],
            refit_window_days = params["refit_window_days"],
        )
        rd._model         = payload["model"]
        rd._label_map     = payload["label_map"]
        rd._feature_names = payload["feature_names"]
        rd._is_fitted     = True

        logger.info(f"RegimeDetector loaded ← {path} | label_map={rd._label_map}")
        return rd

    def summary(self) -> str:
        """
        Return a human-readable description of the fitted model.

        Returns
        -------
        str
        """
        if not self._is_fitted:
            return (
                "RegimeDetector (unfitted)\n"
                f"  n_regimes={self.n_regimes} | cov={self.covariance_type} | "
                f"n_iter={self.n_iter} | min_hold={self.min_regime_hold}"
            )

        lines = [
            "=" * 60,
            "  RegimeDetector — Gaussian HMM (fitted)",
            "=" * 60,
            f"  n_regimes       : {self.n_regimes}",
            f"  covariance_type : {self.covariance_type}",
            f"  n_iter          : {self.n_iter}",
            f"  min_regime_hold : {self.min_regime_hold} days",
            f"  converged       : {self._model.monitor_.converged}",
            f"  log-likelihood  : {self._model.monitor_.history[-1]:.4f}",
            "",
            "  Label map (HMM state → regime):",
        ]
        for state, label in sorted(self._label_map.items()):
            mean_ret = self._model.means_[state][
                self._feature_names.index(_LOG_RET_FEATURE)
                if _LOG_RET_FEATURE in self._feature_names else 0
            ]
            lines.append(f"    state {state} → {label}  (mean log-ret ≈ {mean_ret:.4f})")

        lines.append("=" * 60)
        return "\n".join(lines)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _auto_label(self, X_norm: pd.DataFrame) -> dict[int, str]:
        """
        Map HMM integer states → regime names based on mean spy_log_ret.

        Sorting logic:
          - State with highest mean spy_log_ret  → 'Bull'
          - State with lowest  mean spy_log_ret  → 'Bear'
          - Remaining state(s)                   → 'Sideways'

        Falls back to the first feature column if spy_log_ret is absent.
        """
        if _LOG_RET_FEATURE in X_norm.columns:
            feat_idx = X_norm.columns.tolist().index(_LOG_RET_FEATURE)
        else:
            feat_idx = 0
            logger.warning(
                f"'{_LOG_RET_FEATURE}' not in features — using column 0 "
                "for auto-labelling. Label assignment may be inaccurate."
            )

        # Mean of the target feature per HMM state
        mean_rets = {s: self._model.means_[s][feat_idx]
                     for s in range(self.n_regimes)}

        sorted_states = sorted(mean_rets.items(), key=lambda kv: kv[1])
        # sorted_states[0] = (state, lowest_mean)  → Bear
        # sorted_states[-1]= (state, highest_mean) → Bull

        label_map: dict[int, str] = {}

        if self.n_regimes == 1:
            label_map[sorted_states[0][0]] = "Bull"
        elif self.n_regimes == 2:
            label_map[sorted_states[0][0]]  = "Bear"
            label_map[sorted_states[-1][0]] = "Bull"
        else:
            label_map[sorted_states[0][0]]  = "Bear"
            label_map[sorted_states[-1][0]] = "Bull"
            # Middle states → Sideways (handles n_regimes > 3 gracefully)
            sideways_names = ["Sideways", "Sideways-2", "Sideways-3"]
            for i, (state, _) in enumerate(sorted_states[1:-1]):
                label_map[state] = sideways_names[min(i, len(sideways_names) - 1)]

        return label_map

    def _apply_min_hold(self, labels: pd.Series) -> pd.Series:
        """
        Smooth out transient regime blips shorter than min_regime_hold days.

        Algorithm: scan the label sequence and replace any contiguous run
        shorter than min_regime_hold with the previous regime (forward-fill
        from the last long-enough run).

        Parameters
        ----------
        labels : pd.Series[str]

        Returns
        -------
        pd.Series[str]  — smoothed labels, same index.
        """
        if self.min_regime_hold <= 1:
            return labels

        result  = labels.copy()
        values  = labels.values
        n       = len(values)
        i       = 0
        prev_label = values[0]  # first label always kept

        while i < n:
            j = i + 1
            while j < n and values[j] == values[i]:
                j += 1
            run_len = j - i
            if run_len < self.min_regime_hold and i > 0:
                # Replace this short run with the previous stable label
                result.iloc[i:j] = prev_label
            else:
                prev_label = values[i]
            i = j

        return result

    def _validate_input(
        self,
        X: pd.DataFrame,
        require_fitted: bool = True,
    ) -> None:
        """Raise informative errors for bad inputs."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"X must be a pd.DataFrame, got {type(X).__name__}."
            )
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError(
                "X must have a DatetimeIndex. "
                "Pass the output of FeatureEngineer.fit_transform()."
            )
        if X.isna().any().any():
            n_nan = int(X.isna().sum().sum())
            raise ValueError(
                f"X contains {n_nan} NaN value(s). "
                "Run FeatureEngineer.fit_transform() which drops NaN rows."
            )
        if len(X) < 2:
            raise ValueError(
                f"X has only {len(X)} row(s). Need at least 2 for HMM training."
            )
        if require_fitted and not self._is_fitted:
            raise RuntimeError(
                "RegimeDetector is not fitted. Call fit() first."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Quick-run entry point: python -m src.regime_detector
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("  Market Regime Detector — Phase 3: HMM Regime Detection")
    print("=" * 60 + "\n")

    # ── Synthetic data (mirrors test_feature_engineer fixtures) ──────────────
    from src.feature_engineer import FeatureEngineer

    n   = 800
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2010-01-04", periods=n)

    def _gbm(S0=100.0, mu=0.0001, sigma=0.012):
        r = rng.normal(mu, sigma, n)
        return S0 * np.exp(np.cumsum(r))

    data = pd.DataFrame({
        "spy":           _gbm(100.0),
        "qqq":           _gbm(200.0),
        "tlt":           _gbm(90.0,  mu=-0.00005, sigma=0.010),
        "uso":           _gbm(30.0,  mu=0.0,      sigma=0.020),
        "gld":           _gbm(150.0, mu=0.00005,  sigma=0.008),
        "vix":           rng.uniform(10, 40, n),
        "yield_10y":     rng.uniform(1.5, 4.5, n),
        "yield_2y":      rng.uniform(0.5, 4.0, n),
        "cpi":           272.0 + np.cumsum(rng.uniform(0.0, 0.3, n)),
        "unemployment":  rng.uniform(3.5, 10.0, n),
    }, index=dates)
    data.index.name  = "date"
    data["yield_spread"] = data["yield_10y"] - data["yield_2y"]

    fe    = FeatureEngineer()
    X_raw = fe.compute_features(data)
    X_norm = fe.fit_transform(X_raw)

    print(f"Feature matrix  : {X_norm.shape}")

    # ── Train HMM ────────────────────────────────────────────────────────────
    rd = RegimeDetector(n_regimes=3, n_iter=200)
    rd.fit(X_norm)

    print(rd.summary())

    # ── Predict & display ────────────────────────────────────────────────────
    regimes   = rd.predict(X_norm)
    proba     = rd.predict_proba(X_norm)
    stats     = rd.get_regime_stats(X_norm, X_raw)

    print("\n[LAST 10 REGIME LABELS]")
    print(regimes.tail(10).to_string())

    print("\n[LAST 5 REGIME PROBABILITIES]")
    print(proba.tail(5).round(3).to_string())

    print("\n[REGIME STATS — meta only]")
    if ("meta", "count") in stats.columns:
        print(stats["meta"].to_string())

    print(f"\n✅ Phase 3 complete. Regime distribution:")
    print(regimes.value_counts().to_string())
    print("─" * 60 + "\n")
