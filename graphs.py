# import pandas as pd
# import seaborn as sns
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import loadData


def missingBarChart():
    df = loadData()
    df = df.apply(lambda s: s.str.strip() if s.dtype == "object" else s)
    df = df.replace("?", np.nan)
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    pct = missing / len(df) * 100
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(missing.index, missing.values, color="#4C72B0")

    # Label each bar with its count and percentage
    for bar, count, p in zip(bars, missing.values, pct.values):
        ax.text(
            bar.get_width() + max(missing) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,} ({p:.1f}%)",
            va="center",
            fontsize=10,
        )

    ax.set_xlabel("Number of missing values")
    ax.set_title("Missing values per column — UCI Adult dataset")
    ax.set_xlim(0, max(missing) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    output_dir = Path("output/beforeProcessing")
    plt.savefig(output_dir / "missing_values.png", dpi=200)
    # plt.show()


def histo(df):
    plt.hist(df["income"].astype(int), bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1], ["True", "False"])
    plt.show()
