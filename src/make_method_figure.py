#!/usr/bin/env python3
"""Draw the CalibFreq-Patch pipeline as a publication-ready vector figure."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "input": "#E9EEF3",
    "semantic": "#DCEAF7",
    "frequency": "#F7E7D3",
    "calibration": "#E5E1F2",
    "fusion": "#DDEFD9",
    "border": "#263238",
}


def box(axis, xy, width, height, text, color, fontsize=5.2, weight="normal"):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.015",
        facecolor=color,
        edgecolor=COLORS["border"],
        linewidth=0.75,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)
    return patch


def arrow(axis, start, end, style="-"):
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7,
            linewidth=0.75,
            linestyle=style,
            color=COLORS["border"],
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    out = Path("figures")
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.85, 2.35))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.03)
    ax.axis("off")

    box(ax, (0.01, 0.41), 0.105, 0.18, "Input\n224 × 224", COLORS["input"], weight="bold")

    box(ax, (0.16, 0.69), 0.14, 0.17, "Frozen\nResNet-18", COLORS["semantic"], weight="bold")
    box(ax, (0.34, 0.69), 0.14, 0.17, "96-D\npatches", COLORS["semantic"])
    box(ax, (0.52, 0.69), 0.15, 0.17, "Stratified memory\n≤ 1,800 + 3-NN", COLORS["semantic"], fontsize=4.7)
    box(ax, (0.71, 0.69), 0.13, 0.17, "Normal-tail\ncalibration", COLORS["calibration"], fontsize=4.8)

    box(ax, (0.16, 0.19), 0.14, 0.17, "High-pass\nσ = 1, 2; ∇", COLORS["frequency"], weight="bold")
    box(ax, (0.34, 0.19), 0.14, 0.17, "Location-wise\nmedian/MAD", COLORS["frequency"], fontsize=4.8)
    box(ax, (0.52, 0.19), 0.15, 0.17, "Frequency\nmap  F", COLORS["frequency"], weight="bold")
    box(ax, (0.71, 0.19), 0.13, 0.17, "Normal-tail\ncalibration", COLORS["calibration"], fontsize=4.8)

    box(ax, (0.875, 0.44), 0.115, 0.18, "Bounded\nagreement\ngate", COLORS["fusion"], fontsize=4.8, weight="bold")
    box(ax, (0.78, 0.01), 0.21, 0.11, "Map + top-4 image score", COLORS["fusion"], fontsize=4.7)

    arrow(ax, (0.115, 0.52), (0.16, 0.77))
    arrow(ax, (0.115, 0.48), (0.16, 0.28))
    for left, right in ((0.30, 0.34), (0.48, 0.52), (0.67, 0.71)):
        arrow(ax, (left, 0.775), (right, 0.775))
        arrow(ax, (left, 0.275), (right, 0.275))
    arrow(ax, (0.84, 0.775), (0.875, 0.57))
    arrow(ax, (0.84, 0.275), (0.875, 0.49))
    arrow(ax, (0.935, 0.44), (0.90, 0.12))

    ax.text(0.23, 0.91, "semantic patch branch", fontsize=5.0, ha="center", weight="bold", color="#356284")
    ax.text(0.23, 0.12, "local residual branch", fontsize=5.0, ha="center", weight="bold", color="#8B5A2B")
    ax.text(0.775, 0.46, r"$\widetilde D,\widetilde F$", fontsize=5.5, ha="center")
    ax.text(0.49, 0.50, "held-out normals only: branch scaling + decision threshold", fontsize=5.0, ha="center", color="#4A3F6B")
    ax.text(0.01, 0.96, "CalibFreq-Patch", fontsize=6.0, weight="bold", color="#263238")
    ax.text(0.01, 0.02, "No target defect is used for fitting, calibration, or thresholding", fontsize=4.8, style="italic")

    fig.savefig(out / "method_overview.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out / "method_overview.png", dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
