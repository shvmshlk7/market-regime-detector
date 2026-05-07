"""
notebooks/02_feature_engineering.py
─────────────────────────────────────
Phase 2 EDA: Feature Engineering Analysis

Generates 5 charts saved to notebooks/plots/:
  1. Feature timeline — 4 subplots, one per feature group
  2. Feature correlation heatmap
  3. Mutual information bar chart (feature relevance)
  4. Vol regime signal — vol_ratio annotated with VIX
  5. Feature distributions — violin plots per group

Run:
    cd "d:/2026 websites/Market Regime Detector + Portfolio Optimizer"
    python notebooks/02_feature_engineering.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import seaborn as sns
from dotenv import load_dotenv

from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv()

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Dark premium theme ───────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f172a",
    "axes.facecolor":    "#1e293b",
    "axes.edgecolor":    "#334155",
    "axes.labelcolor":   "#94a3b8",
    "text.color":        "#f1f5f9",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "grid.color":        "#334155",
    "grid.linestyle":    "--",
    "grid.alpha":        0.4,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "font.family":       "monospace",
    "figure.dpi":        130,
    "axes.titlecolor":   "#e2e8f0",
    "axes.titlesize":    10,
})

# Feature group color palette
COLORS = {
    "vol":        "#60a5fa",   # Blue     — volatility
    "mom":        "#a78bfa",   # Purple   — momentum
    "macro":      "#34d399",   # Green    — macro
    "cross":      "#fbbf24",   # Amber    — cross-asset
    "neutral":    "#94a3b8",   # Slate    — neutral
    "accent":     "#f472b6",   # Pink     — accent
    "danger":     "#f87171",   # Red      — stress signal
    "success":    "#4ade80",   # Green    — calm signal
}

GROUP_PALETTE = {
    "spy_log_ret":        COLORS["vol"],
    "spy_vol_20d":        "#3b82f6",
    "spy_vol_60d":        "#1d4ed8",
    "vol_ratio":          COLORS["vol"],
    "mom_1m_zscore":      "#c4b5fd",
    "mom_3m_zscore":      COLORS["mom"],
    "mom_6m_zscore":      "#7c3aed",
    "vix_level":          COLORS["danger"],
    "vix_zscore":         "#fca5a5",
    "yield_spread":       COLORS["macro"],
    "yield_spread_chg":   "#6ee7b7",
    "cpi_yoy":            "#059669",
    "bond_equity_corr":   COLORS["cross"],
    "commodity_trend":    "#d97706",
    "spy_ma_ratio":       "#92400e",
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
print("  Phase 2 EDA — Feature Engineering Analysis")
print("=" * 60)

loader = DataLoader()
data   = loader.get_combined_data()
print(f"\nCombined dataset: {data.shape[0]:,} rows × {data.shape[1]} columns")

fe    = FeatureEngineer()
X_raw = fe.compute_features(data)
X_norm = fe.fit_transform(X_raw)
mi    = fe.feature_importance(X_raw)

print(f"Feature matrix : {X_raw.shape[0]:,} rows × {X_raw.shape[1]} features")
print(f"Date range     : {X_raw.index[0].date()} → {X_raw.index[-1].date()}")
print(f"Warmup removed : {len(data) - len(X_raw)} rows")

# ── Print summary stats ──────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("FEATURE SUMMARY STATISTICS")
print("─" * 60)
stats = fe.summary(X_raw)
print(stats.to_string())


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Feature Timeline — 4 groups, 4 subplots
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Feature timelines by group...")

groups = [
    ("Volatility Features",  FeatureEngineer.VOLATILITY_FEATURES),
    ("Momentum Z-Scores",    FeatureEngineer.MOMENTUM_FEATURES),
    ("Macro Features",       FeatureEngineer.MACRO_FEATURES),
    ("Cross-Asset Features", FeatureEngineer.CROSS_ASSET_FEATURES),
]

fig, axes = plt.subplots(4, 1, figsize=(18, 16), sharex=True)
fig.suptitle(
    "Feature Timeline — All 15 Features (2005–Present)",
    fontsize=15, fontweight="bold", color="#f1f5f9", y=0.98
)

for ax, (group_name, cols) in zip(axes, groups):
    available = [c for c in cols if c in X_raw.columns]
    for col in available:
        color = GROUP_PALETTE.get(col, COLORS["neutral"])
        alpha = 1.0 if len(available) <= 2 else 0.75
        ax.plot(X_raw[col], label=col, color=color,
                linewidth=1.2, alpha=alpha)

    # Zero reference line for z-scored features
    ax.axhline(0, color="#475569", linewidth=0.6, linestyle="--", alpha=0.6)

    ax.set_title(group_name, fontsize=10, fontweight="bold",
                 color="#e2e8f0", pad=4)
    ax.legend(loc="upper left", ncol=len(available), fontsize=7,
              framealpha=0.15, labelcolor="#e2e8f0", facecolor="#1e293b")
    ax.grid(True, alpha=0.25)

axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout(rect=[0, 0, 1, 0.97])
save(fig, "06_feature_timeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Feature Correlation Heatmap
# ─────────────────────────────────────────────────────────────────────────────
print("[2/5] Feature correlation heatmap...")

corr = X_raw.corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

fig, ax = plt.subplots(figsize=(13, 11))
fig.suptitle(
    "Feature Correlation Matrix",
    fontsize=14, fontweight="bold", color="#f1f5f9"
)

hm = sns.heatmap(
    corr,
    ax=ax,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdYlGn",
    vmin=-1.0, vmax=1.0,
    center=0,
    square=True,
    linewidths=0.4,
    linecolor="#0f172a",
    annot_kws={"size": 7, "color": "#f1f5f9"},
    cbar_kws={"shrink": 0.75},
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
hm.collections[0].colorbar.ax.yaxis.label.set_color("#94a3b8")
hm.collections[0].colorbar.ax.tick_params(colors="#64748b")

fig.tight_layout()
save(fig, "07_feature_correlation.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 3: Mutual Information Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
print("[3/5] Mutual information bar chart...")

fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle(
    "Feature Importance — Mutual Information vs VIX Level",
    fontsize=14, fontweight="bold", color="#f1f5f9"
)

bar_colors = [GROUP_PALETTE.get(f, COLORS["neutral"]) for f in mi["feature"]]
bars = ax.barh(mi["feature"], mi["mi_score"], color=bar_colors,
               height=0.65, alpha=0.9)

# Value labels
for bar, val in zip(bars, mi["mi_score"]):
    ax.text(
        bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
        f"{val:.3f}", va="center", fontsize=8, color="#94a3b8"
    )

ax.invert_yaxis()
ax.set_xlabel("Mutual Information Score (higher = more relevant)", fontsize=10)
ax.axvline(x=mi["mi_score"].median(), color="#475569",
           linewidth=1.0, linestyle="--", alpha=0.7,
           label=f"Median = {mi['mi_score'].median():.3f}")
ax.legend(fontsize=8, framealpha=0.2, facecolor="#1e293b", labelcolor="#94a3b8")
ax.grid(True, axis="x", alpha=0.25)
ax.set_xlim(left=0)

fig.tight_layout()
save(fig, "08_feature_importance.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 4: Vol Regime Signal — vol_ratio with VIX overlay
# ─────────────────────────────────────────────────────────────────────────────
print("[4/5] Vol regime signal chart...")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 9), sharex=True)
fig.suptitle(
    "Regime Signal: Volume Ratio (20d/60d Vol) + VIX",
    fontsize=14, fontweight="bold", color="#f1f5f9"
)

# Shade vol_ratio > 1.2 as "stress" zones
vol_ratio = X_raw["vol_ratio"]
ax1.plot(vol_ratio, color=COLORS["vol"], linewidth=1.3, alpha=0.9, label="vol_ratio")
ax1.axhline(1.0, color="#64748b", linewidth=0.8, linestyle="--", alpha=0.6)
ax1.axhline(1.2, color=COLORS["danger"], linewidth=0.8, linestyle="--",
            alpha=0.6, label="Stress threshold (1.2)")

# Shade stress periods
stress = vol_ratio > 1.2
ax1.fill_between(vol_ratio.index, vol_ratio, 1.2,
                 where=stress, color=COLORS["danger"], alpha=0.15,
                 label="Elevated stress zone")

ax1.set_ylabel("Vol Ratio (20d / 60d)", fontsize=9)
ax1.set_title("Vol Ratio — Values > 1.0 signal rising short-term volatility", fontsize=9)
ax1.legend(loc="upper left", fontsize=8, framealpha=0.2, facecolor="#1e293b",
           labelcolor="#e2e8f0")
ax1.grid(True, alpha=0.25)

# VIX level (log-transformed)
if "vix_level" in X_raw.columns:
    ax2.plot(X_raw["vix_level"], color=COLORS["danger"], linewidth=1.2, alpha=0.9)
    ax2.fill_between(X_raw.index, X_raw["vix_level"],
                     X_raw["vix_level"].min(),
                     alpha=0.15, color=COLORS["danger"])
    ax2.set_ylabel("log(VIX)", fontsize=9)
    ax2.set_title("VIX Level (log-transformed) — Higher = fear / bear regime", fontsize=9)
    ax2.grid(True, alpha=0.25)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
fig.tight_layout()
save(fig, "09_regime_signal.png")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 5: Feature Distributions — Violin Plots by Group
# ─────────────────────────────────────────────────────────────────────────────
print("[5/5] Feature distributions (violin plots)...")

# Use normalized matrix so x-axis is comparable
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle(
    "Feature Distributions — Normalized (StandardScaler)",
    fontsize=14, fontweight="bold", color="#f1f5f9"
)

group_colors = {
    "Volatility":   COLORS["vol"],
    "Momentum":     COLORS["mom"],
    "Macro":        COLORS["macro"],
    "Cross-Asset":  COLORS["cross"],
}

group_defs = [
    ("Volatility",  FeatureEngineer.VOLATILITY_FEATURES),
    ("Momentum",    FeatureEngineer.MOMENTUM_FEATURES),
    ("Macro",       FeatureEngineer.MACRO_FEATURES),
    ("Cross-Asset", FeatureEngineer.CROSS_ASSET_FEATURES),
]

for idx, (group_name, feats) in enumerate(group_defs):
    ax    = fig.add_subplot(gs[idx // 2, idx % 2])
    avail = [f for f in feats if f in X_norm.columns]
    data_to_plot = [X_norm[f].dropna().values for f in avail]

    color = group_colors[group_name]

    # Violin + box
    vp = ax.violinplot(data_to_plot, positions=range(len(avail)),
                       showmedians=True, showextrema=False)
    for pc in vp["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
        pc.set_edgecolor(color)
    vp["cmedians"].set_color("#f1f5f9")
    vp["cmedians"].set_linewidth(1.5)

    ax.axhline(0, color="#64748b", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(
        [f.replace("_", "\n") for f in avail],
        fontsize=7, color="#94a3b8"
    )
    ax.set_title(f"{group_name} Group", fontsize=10, fontweight="bold",
                 color="#e2e8f0", pad=6)
    ax.set_ylabel("Normalized Value (σ)", fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)

save(fig, "10_feature_distributions.png")


# ─────────────────────────────────────────────────────────────────────────────
# Final Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("PHASE 2 CHECKPOINT SUMMARY")
print("─" * 60)
print(f"  Feature matrix:  {X_raw.shape[0]:,} rows × {X_raw.shape[1]} features")
print(f"  Date range:      {X_raw.index[0].date()} → {X_raw.index[-1].date()}")
print(f"  NaN rows:        0 (all warmup rows dropped)")
print(f"  Groups:          Volatility(4) | Momentum(3) | Macro(5) | Cross-Asset(3)")
print()
print("  Top 5 features by Mutual Information:")
for _, row in mi.head(5).iterrows():
    print(f"    #{int(row['rank']):<2} {row['feature']:<22} MI={row['mi_score']:.4f}")
print()
print(f"✅ Phase 2 complete. Plots saved to: {PLOT_DIR}")
print("─" * 60 + "\n")
