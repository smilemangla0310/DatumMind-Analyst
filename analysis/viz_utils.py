"""
Visualization helpers — dark theme, chart export, and utility functions.
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import io


# ── DatumMind Dark Theme Color Palette ────────────────────────────────────────
DATUMMIND_COLORS = [
    "#6366F1",  # Indigo (primary)
    "#22D3EE",  # Cyan
    "#F59E0B",  # Amber
    "#EF4444",  # Red
    "#10B981",  # Emerald
    "#8B5CF6",  # Violet
    "#F97316",  # Orange
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#A78BFA",  # Light Violet
]


def set_plot_style():
    """Configure dark-theme plot style matching the dashboard."""
    sns.set_theme(style="darkgrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.figsize": (10, 5),
        "figure.dpi": 100,
        "figure.facecolor": "#1A1D29",
        "axes.facecolor": "#252836",
        "axes.edgecolor": "#2A2D3E",
        "axes.labelcolor": "#94A3B8",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.titlecolor": "#F1F5F9",
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "#2A2D3E",
        "grid.alpha": 0.4,
        "text.color": "#94A3B8",
        "xtick.color": "#64748B",
        "ytick.color": "#64748B",
        "legend.facecolor": "#1A1D29",
        "legend.edgecolor": "#2A2D3E",
        "legend.framealpha": 0.9,
        "legend.labelcolor": "#94A3B8",
        "font.family": "sans-serif",
        "savefig.facecolor": "#1A1D29",
        "savefig.edgecolor": "none",
    })
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=DATUMMIND_COLORS)


def save_figure_to_bytes(fig) -> bytes:
    """Convert a matplotlib figure to PNG bytes for download."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor="#1A1D29", edgecolor="none")
    buf.seek(0)
    return buf.getvalue()


def close_all_figures():
    """Close all open matplotlib figures to prevent memory leaks."""
    plt.close("all")
