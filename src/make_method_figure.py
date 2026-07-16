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


def box(axis, xy, width, height, text, color, fontsize=8.0, weight="normal"):
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
    fig, ax = plt.subplots(figsize=(4.85, 2.50))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.05, 1.03)
    ax.axis("off")

    box(ax, (0.01, 0.42), 0.11, 0.16, "Input\n224 px", COLORS["input"], weight="bold")

    box(ax, (0.16, 0.68), 0.27, 0.17, "ResNet-18\n96-D patches", COLORS["semantic"], weight="bold")
    box(ax, (0.48, 0.68), 0.28, 0.17, "≤1,800 memory\n3-NN + tail scale", COLORS["calibration"])

    box(ax, (0.16, 0.17), 0.27, 0.19, "High-pass\nr₁, r₂, g\nmedian/MAD", COLORS["frequency"], weight="bold")
    box(ax, (0.48, 0.18), 0.28, 0.17, "Frequency score F\n+ tail scale", COLORS["calibration"])

    box(ax, (0.82, 0.41), 0.16, 0.18, "Bounded\ngate", COLORS["fusion"], weight="bold")
    box(ax, (0.76, 0.02), 0.22, 0.10, "Fused map\nTop-4 score", COLORS["fusion"])

    arrow(ax, (0.12, 0.52), (0.16, 0.76))
    arrow(ax, (0.12, 0.48), (0.16, 0.27))
    arrow(ax, (0.43, 0.765), (0.48, 0.765))
    arrow(ax, (0.43, 0.265), (0.48, 0.265))
    arrow(ax, (0.76, 0.765), (0.82, 0.55))
    arrow(ax, (0.76, 0.265), (0.82, 0.46))
    arrow(ax, (0.90, 0.41), (0.87, 0.12))

    ax.text(0.25, 0.90, "semantic patch branch", fontsize=8.0, ha="center", weight="bold", color="#356284")
    ax.text(0.25, 0.11, "local residual branch", fontsize=8.0, ha="center", weight="bold", color="#8B5A2B")
    ax.text(0.47, 0.50, "Held-out normals only:\nscales + threshold", fontsize=8.0, ha="center", va="center", color="#4A3F6B")
    ax.text(0.01, 0.96, "CalibFreq-Patch", fontsize=8.5, weight="bold", color="#263238")

    fig.savefig(out / "method_overview.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out / "method_overview.png", dpi=1200, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
