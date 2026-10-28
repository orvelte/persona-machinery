"""Figure style (spec Deliverables): Liberation Serif, inward ticks on all sides, faint horizontal grid, fixed palette, italic subtitle."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PALETTE = ["#3B73C8", "#E07B39", "#4CAF7D", "#E5A93B", "#D9769C", "#8E6BBF", "#3BA7A7"]
HATCHES = ["", "///", "...", "xxx", "\\\\\\", "+++"]
LINESTYLES = ["-", "--", ":", "-."]


def setup():
    for f in (REPO / "figures/fonts").glob("LiberationSerif*.ttf"):
        fm.fontManager.addfont(str(f))
    matplotlib.rcParams.update({
        "font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.minor.visible": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#DDDDDD", "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "axes.spines.top": True, "axes.spines.right": True, "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
        "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
    })


def title(ax, main: str, sub: str):
    ax.set_title(main, loc="left", fontweight="bold")
    ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=8.5, style="italic", va="bottom", ha="left", color="#444444")
    ax.title.set_position((0.0, 1.06))


def save(fig, name: str):
    out = REPO / "figures" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
