"""
notebooks/01_eda.py
────────────────────
Phase 1 EDA: Exploratory Data Analysis

Generates the following charts and saves them to notebooks/plots/:
  1. Price history for all ETFs (normalized to 100)
  2. Correlation heatmap across all ETFs
  3. Rolling 20-day volatility for each ETF
  4. Macro indicators over time (VIX, yield spread, CPI, unemployment)
  5. Missing value heatmap

Run:
    cd "d:/2026 websites/Market Regime Detector + Portfolio Optimizer"
    python notebooks/01_eda.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from dotenv import load_dotenv

from src.data_loader import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# Premium dark theme for all charts
plt.rcParams.update({
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#1e293b",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#94a3b8",
    "text.color":        "#f1f5f9",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "grid.color":        "#1e293b",
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "monospace",
    "figure.dpi":        130,
})

# ETF colors — rich, distinct palette
ETF_COLORS = {
    "spy": "#60a5fa", "qqq": "#a78bfa", "iwm": "#34d399",
    "gld": "#fbbf24", "tlt": "#fb923c", "lqd": "#f472b6",
    "vnq": "#2dd4bf", "uso": "#f87171", "efa": "#c084fc",
    "eem": "#4ade80",
}

MACRO_COLORS = {
    "vix":          "#f87171",
    "yield_10y":    "#60a5fa",
    "yield_2y":     "#a78bfa",
    "yield_spread": "#fbbf24",
    "cpi":          "#34d399",
    "unemployment": "#fb923c",
}


def save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  ✓ Saved: {path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  Phase 1 EDA — Market Regime Detector")
print("=" * 60)

loader   = DataLoader()
etf_data = loader.fetch_etf_data()
etf_data.columns = [c.lower() for c in etf_data.columns]

try:
    macro_data = loader.fetch_macro_data()
    has_macro  = not macro_data.empty
except Exception:
    macro_data = pd.DataFrame()
    has_macro  = False

log_returns = loader.get_returns(etf_data)

print(f"\nETF data loaded:   {etf_data.shape[0]:,} rows × {etf_data.shape[1]} tickers")
print(f"Date range:        {etf_data.index[0].date()} → {etf_data.index[-1].date()}")
print(f"Macro data loaded: {has_macro}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Normalized Price History (Base 100)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Plotting normalized price history...")

fig, ax = plt.subplots(figsize=(16, 7))
fig.suptitle(
    "ETF Price History — Normalized to 100 (2005–Present)",
    fontsize=14, fontweight="bold", color="#f1f5f9", y=0.98
)

for ticker in etf_data.columns:
    series     = etf_data[ticker].dropna()
    normalized = series / series.iloc[0] * 100
    color      = ETF_COLORS.get(ticker, "#94a3b8")
    ax.plot(normalized, label=ticker.upper(), color=color, linewidth=1.4, alpha=0.85)

# Shade known crash periods
crash_periods = [
    ("2007-10-01", "2009-03-31", "GFC 2008"),
    ("2020-02-20", "2020-03-23", "COVID Crash"),
    ("2022-01-01", "2022-12-31", "2022 Bear"),
]
for start, end, label in crash_periods:
    try:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                   alpha=0.12, color="#ef4444", zorder=0)
        mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
        ax.text(mid, ax.get_ylim()[1] * 0.95, label,
                ha="center", fontsize=8, color="#fca5a5", alpha=0.8)
    except Exception:
        pass

ax.set_xlabel("Date", fontsize=10)
ax.set_ylabel("Normalized Price (Base = 100)", fontsize=10)
ax.legend(loc="upper left", ncol=2, fontsize=8, framealpha=0.2,
          labelcolor="#e2e8f0", facecolor="#1e293b")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
save(fig, "01_price_history.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("[2/5] Plotting correlation heatmap...")

corr = log_returns.corr()
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle(
    "ETF Return Correlation Matrix",
    fontsize=14, fontweight="bold", color="#f1f5f9"
)

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Hide upper triangle
sns.heatmap(
    corr,
    ax=ax,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1, vmax=1,
    center=0,
    square=True,
    linewidths=0.5,
    linecolor="#0f172a",
    annot_kws={"size": 9},
    cbar_kws={"shrink": 0.8},
)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
fig.tight_layout()
save(fig, "02_correlation_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Rolling 20-Day Volatility
# ─────────────────────────────────────────────────────────────────────────────
print("[3/5] Plotting rolling volatility...")

rolling_vol = log_returns.rolling(20).std() * np.sqrt(252) * 100  # Annualized %

fig, ax = plt.subplots(figsize=(16, 6))
fig.suptitle(
    "Rolling 20-Day Annualized Volatility (%)",
    fontsize=14, fontweight="bold", color="#f1f5f9"
)

# Highlight SPY vol prominently, others as background
for ticker in rolling_vol.columns:
    if ticker == "spy":
        ax.plot(rolling_vol[ticker], label="SPY", color="#60a5fa",
                linewidth=2.2, alpha=0.95, zorder=5)
    else:
        color = ETF_COLORS.get(ticker, "#94a3b8")
        ax.plot(rolling_vol[ticker], label=ticker.upper(), color=color,
                linewidth=0.9, alpha=0.5)

ax.set_xlabel("Date")
ax.set_ylabel("Annualized Volatility (%)")
ax.legend(loc="upper left", ncol=3, fontsize=7, framealpha=0.2,
          labelcolor="#e2e8f0", facecolor="#1e293b")
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
save(fig, "03_rolling_volatility.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Macro Indicators
# ─────────────────────────────────────────────────────────────────────────────
print("[4/5] Plotting macro indicators...")

if has_macro:
    available_cols = [c for c in MACRO_COLORS if c in macro_data.columns]
    n = len(available_cols)
    if n > 0:
        ncols = 2
        nrows = (n + 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        fig.suptitle(
            "Macro Indicators (FRED Data)",
            fontsize=14, fontweight="bold", color="#f1f5f9"
        )
        axes = axes.flatten() if n > 1 else [axes]

        titles = {
            "vix":          "VIX (Fear Index)",
            "yield_10y":    "10-Year Treasury Yield (%)",
            "yield_2y":     "2-Year Treasury Yield (%)",
            "yield_spread": "Yield Spread (10Y - 2Y) %",
            "cpi":          "CPI — All Urban Consumers",
            "unemployment": "Unemployment Rate (%)",
        }

        for i, col in enumerate(available_cols):
            ax    = axes[i]
            color = MACRO_COLORS[col]
            ax.plot(macro_data[col].dropna(), color=color, linewidth=1.4)
            ax.fill_between(macro_data[col].dropna().index,
                            macro_data[col].dropna(), alpha=0.15, color=color)
            ax.set_title(titles.get(col, col), fontsize=10, color="#e2e8f0")
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

            # Shade yield_spread when negative (inverted = recession signal)
            if col == "yield_spread":
                neg = macro_data[col].dropna()
                ax.fill_between(neg.index, neg, 0,
                                where=(neg < 0),
                                color="#ef4444", alpha=0.25,
                                label="Inverted (recession signal)")
                ax.axhline(0, color="#64748b", linewidth=0.8, linestyle="--")
                ax.legend(fontsize=7, framealpha=0.2, facecolor="#1e293b",
                          labelcolor="#fca5a5")

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        save(fig, "04_macro_indicators.png")
else:
    print("  ⚠ Skipping macro plots — FRED data unavailable")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Missing Value Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("[5/5] Plotting missing value heatmap...")

# Sample every 10th row for readability
sample = etf_data.iloc[::10].isna()

fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle(
    "Missing Values Heatmap (sampled every 10 trading days)",
    fontsize=13, fontweight="bold", color="#f1f5f9"
)

sns.heatmap(
    sample.T,
    ax=ax,
    cmap=["#1e293b", "#ef4444"],  # Dark = present, Red = missing
    cbar=False,
    yticklabels=True,
    xticklabels=False,
    linewidths=0,
)
ax.set_ylabel("Ticker", fontsize=10)
ax.set_xlabel("Time →", fontsize=10)
fig.tight_layout()
save(fig, "05_missing_values.png")


# ─────────────────────────────────────────────────────────────────────────────
# Final: Run Validation + Print Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("DATA VALIDATION")
print("─" * 60)
report = loader.validate_data(etf_data)
print(f"\n  Shape:            {report['shape']}")
print(f"  Date range:       {report['date_range']}")
print(f"  Trading days:     {report['trading_days']:,}")
print(f"  Years of history: {report['years_of_history']}")
print(f"  Status:           {'✅ PASSED' if report['passed'] else '⚠️  WARNINGS'}")

if report["suspicious_columns"]:
    print(f"  ⚠ Suspicious:   {report['suspicious_columns']}")

# Summary stats for ETFs
print("\n" + "─" * 60)
print("ETF SUMMARY STATISTICS (Annualized Returns & Volatility)")
print("─" * 60)
ann_returns = log_returns.mean() * 252 * 100
ann_vol     = log_returns.std() * np.sqrt(252) * 100
sharpe      = ann_returns / ann_vol

summary = pd.DataFrame({
    "Ann. Return (%)": ann_returns.round(2),
    "Ann. Vol (%)":    ann_vol.round(2),
    "Sharpe (approx)": sharpe.round(2),
})
print(summary.to_string())

print(f"\n✅ EDA complete. Plots saved to: {PLOT_DIR}")
print("─" * 60 + "\n")
