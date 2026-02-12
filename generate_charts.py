#!/usr/bin/env python3
"""Generate benchmark bar chart images for README."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

# Benchmark data
samples = ["small\n(3.0 KB)", "medium\n(50.1 KB)", "large\n(100.5 KB)"]
originals = [3103, 51317, 102863]

gzip_bytes = [1453, 20101, 39965]
xz_bytes = [1559, 18788, 36332]
zip_bytes = [1549, 20194, 40061]
nacrith_bytes = [424, 7602, 15886]

gzip_ratio = [g/o*100 for g, o in zip(gzip_bytes, originals)]
xz_ratio = [x/o*100 for x, o in zip(xz_bytes, originals)]
zip_ratio = [z/o*100 for z, o in zip(zip_bytes, originals)]
nacrith_ratio = [n/o*100 for n, o in zip(nacrith_bytes, originals)]

# Colors
C_GZIP = "#6C8EBF"
C_XZ = "#82B366"
C_ZIP = "#D6B656"
C_NACRITH = "#E04040"
BG_COLOR = "#0D1117"
TEXT_COLOR = "#C9D1D9"
GRID_COLOR = "#21262D"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "font.family": "sans-serif",
    "font.size": 13,
})

# ── Chart 1: Compression Ratio ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 6.5))

x = np.arange(len(samples))
w = 0.19

bars_gzip = ax.bar(x - 1.5*w, gzip_ratio, w, label="gzip", color=C_GZIP, edgecolor=C_GZIP, linewidth=0.5, zorder=3)
bars_xz = ax.bar(x - 0.5*w, xz_ratio, w, label="xz", color=C_XZ, edgecolor=C_XZ, linewidth=0.5, zorder=3)
bars_zip = ax.bar(x + 0.5*w, zip_ratio, w, label="zip", color=C_ZIP, edgecolor=C_ZIP, linewidth=0.5, zorder=3)
bars_ncr = ax.bar(x + 1.5*w, nacrith_ratio, w, label="Nacrith", color=C_NACRITH, edgecolor=C_NACRITH, linewidth=0.5, zorder=3)

# Labels on bars
for bars in [bars_gzip, bars_xz, bars_zip, bars_ncr]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 2.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold", color=TEXT_COLOR)

# 100% reference line
ax.axhline(y=100, color="#F0883E", linestyle="--", linewidth=1, alpha=0.6, zorder=2)
ax.text(len(samples) - 0.55, 103, "original size", fontsize=9, color="#F0883E", alpha=0.8, ha="right")

ax.set_ylabel("Compressed / Original (%)", fontsize=13)
ax.set_title("Compression Ratio Comparison  (lower = better)", fontsize=16, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(samples, fontsize=11)
ax.set_ylim(0, max(max(gzip_ratio), max(xz_ratio), max(zip_ratio)) * 1.18)
ax.legend(loc="upper right", fontsize=12, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig("/home/st4ck/compressor2/assets/compression_ratio.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: assets/compression_ratio.png")

# ── Chart 2: Compressed Size (bytes) ────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)

sample_names = ["small (3.0 KB)", "medium (50.1 KB)", "large (100.5 KB)"]
all_bytes = list(zip(gzip_bytes, xz_bytes, zip_bytes, nacrith_bytes))
labels = ["gzip", "xz", "zip", "Nacrith"]
colors = [C_GZIP, C_XZ, C_ZIP, C_NACRITH]

for i, (ax, title, vals) in enumerate(zip(axes, sample_names, all_bytes)):
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, vals, color=colors, edgecolor=[c for c in colors], linewidth=0.5, height=0.6, zorder=3)

    for bar, val in zip(bars, vals):
        if val >= 1024:
            label = f"{val/1024:.1f}K"
        else:
            label = f"{val}"
        ax.text(val + max(vals)*0.03, bar.get_y() + bar.get_height()/2,
                label, ha="left", va="center", fontsize=10, fontweight="bold", color=TEXT_COLOR)

    # Original size reference
    ax.axvline(x=originals[i], color="#F0883E", linestyle="--", linewidth=1.2, alpha=0.7, zorder=2)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(0, max(max(vals), originals[i]) * 1.4)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.invert_yaxis()

fig.suptitle("Compressed Size  (orange dashed = original size)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig("/home/st4ck/compressor2/assets/compressed_size.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: assets/compressed_size.png")

# ── Chart 3: Space Savings % ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(16, 6.5))

gzip_savings = [(1 - g/o)*100 for g, o in zip(gzip_bytes, originals)]
xz_savings = [(1 - x/o)*100 for x, o in zip(xz_bytes, originals)]
zip_savings = [(1 - z/o)*100 for z, o in zip(zip_bytes, originals)]
nacrith_savings = [(1 - n/o)*100 for n, o in zip(nacrith_bytes, originals)]

x = np.arange(len(samples))
w = 0.19

ax.bar(x - 1.5*w, gzip_savings, w, label="gzip", color=C_GZIP, zorder=3)
ax.bar(x - 0.5*w, xz_savings, w, label="xz", color=C_XZ, zorder=3)
ax.bar(x + 0.5*w, zip_savings, w, label="zip", color=C_ZIP, zorder=3)
bars_ncr = ax.bar(x + 1.5*w, nacrith_savings, w, label="Nacrith", color=C_NACRITH, zorder=3)

# Highlight Nacrith values
for bar, val in zip(bars_ncr, nacrith_savings):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold", color=C_NACRITH)

ax.axhline(y=0, color=TEXT_COLOR, linewidth=0.8, alpha=0.5, zorder=2)

ax.set_ylabel("Space Savings (%)", fontsize=13)
ax.set_title("Space Savings vs Original  (higher = better, negative = file grew)", fontsize=16, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(samples, fontsize=11)
ax.legend(loc="lower right", fontsize=12, facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor=TEXT_COLOR)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.set_axisbelow(True)

plt.tight_layout()
fig.savefig("/home/st4ck/compressor2/assets/space_savings.png", dpi=180, bbox_inches="tight")
plt.close()
print("Saved: assets/space_savings.png")

print("\nAll charts generated!")
