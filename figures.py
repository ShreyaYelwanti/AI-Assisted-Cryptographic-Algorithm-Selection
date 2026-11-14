"""Matplotlib figures that look like the research notebook's (used by the website and the evidence script)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from core.config import CIPHERS  # noqa: E402


def fig_throughput(summary: pd.DataFrame):
    """Figure 1: encrypt + decrypt throughput by cipher (mean +/- std, log x axis)."""
    fig = plt.figure(figsize=(7, 4.5))
    for name in CIPHERS:
        d = summary[summary["cipher"] == name]
        plt.errorbar(d["size"], d["thr_mean"], yerr=d["thr_std"], marker="o", capsize=3, label=name)
    plt.xscale("log")
    plt.xlabel("Payload size (bytes)")
    plt.ylabel("Throughput (MB/s)")
    plt.title("Encrypt + decrypt throughput by cipher")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def fig_confusion(cm: np.ndarray):
    """Figure 2: Random Forest confusion matrix."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(CIPHERS)), CIPHERS, rotation=20, ha="right")
    ax.set_yticks(range(len(CIPHERS)), CIPHERS)
    top = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{int(cm[i, j])}", ha="center", va="center",
                    color="white" if cm[i, j] > top / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Optimal (label)")
    ax.set_title("Random Forest confusion matrix")
    fig.tight_layout()
    return fig


def fig_importance(imp: pd.Series):
    """Figure 3: Random Forest feature importance."""
    fig = plt.figure(figsize=(6.5, 4.5))
    imp.sort_values().plot.barh()
    plt.xlabel("Importance")
    plt.title("Random Forest feature importance")
    plt.tight_layout()
    return fig


def fig_cost_by_network(by_net: pd.DataFrame):
    """Figure 4: mean normalised cost by network type for each selector."""
    cols = [c for c in by_net.columns if c != "RF + security filter"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    by_net[cols].plot.bar(ax=ax)
    ax.set_ylabel("Mean normalized cost (1.0 = optimal)")
    ax.set_xlabel("Network type")
    ax.set_title("Selector cost by network (new seed)")
    plt.setp(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    return fig


def fig_confidence(proba: pd.Series, chosen: str):
    """Small horizontal bar chart of the AI's confidence per cipher."""
    fig, ax = plt.subplots(figsize=(5, 2.2))
    vals = (proba * 100).reindex(CIPHERS)
    colors = ["#2563eb" if c == chosen else "#9ca3af" for c in vals.index]
    ax.barh(vals.index[::-1], vals.values[::-1], color=colors[::-1])
    for y, v in enumerate(vals.values[::-1]):
        ax.text(min(v + 1, 88), y, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def fig_factor_importance(imp: pd.DataFrame):
    """Small bar chart of the importance of each user-facing factor."""
    fig, ax = plt.subplots(figsize=(5, 2.2))
    d = imp.sort_values("Importance")
    ax.barh(d["Factor"], d["Importance"], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
