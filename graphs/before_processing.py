from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import loadData

# Creates a bar chart to show where data is missing
# Loads the adult csv
# Converts the missing value to '?'
# Counts the number of '?' for each column that had some present
# Creates a bar chart using the number of '?' as the Y axis and the X axis in the name of the category
# Saves the chart in 'output/beforeProcessing/'
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
    plt.close()


# Generates a heat map for missing values
# Shows the correlation between missing 'workclass' and 'occupation'
# Their a a correlation where if one's missing the other is missing
def missingHeatmap():
    df = loadData()
    missing_cols = df.columns[df.isna().any()]
    dirty = df[df.isna().any(axis=1)][missing_cols]

    mask = dirty.isna().astype(int)
    mask = mask.sort_values(by=list(missing_cols), ascending=False)

    fig, ax = plt.subplots(figsize=(6,6))

    ax.imshow(mask.values, aspect='auto', cmap='Greys', interpolation='nearest')
    ax.set_xticks(range(len(missing_cols)))
    ax.set_xticklabels(missing_cols)
    ax.set_ylabel(f"Dirty rows ({len(mask):,})")
    ax.set_yticks([])
    ax.set_title('Missing values in dirty rows (black = missing)')

    plt.tight_layout()
    output_dir = Path('output/beforeProcessing')
    plt.savefig(output_dir / 'missing_heatmap.png', dpi=200)
    plt.close()

# Creates a boxplot for capital gains
# Most values are zero, but their are extreme outliers
def capitalBoxPlots():
    df = loadData()
    cols = ['capital-gains', 'capital-loss']

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col in zip(axes, cols):
        groups = [df.loc[df['income'] == label, col] for label in (False, True)]
        ax.boxplot(groups, tick_labels=['<=50K', '>50K'])
        ax.set_yscale('symlog')
        zero_pct = (df[col] == 0).mean() * 100
        ax.set_title(f"{col}\n({zero_pct:.1f}% of rows are 0)")
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Capital gains and losses by income (symlog scale)')
    plt.tight_layout()
    output_dir = Path('output/beforeProcessing')
    plt.savefig(output_dir / 'capital_boxplots.png', dpi=200)
    plt.close()

# Creates a histograph for the amount of people who make atleat 50K (True) vs people who make less (False)
# Show the target value is true only 1/4~ of the time
# Where as its false 3/4~ times
def histo():
    df = loadData()
    plt.hist(df["income"].astype(int), bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1], ["<=50K False", ">50K True"])
    output_dir = Path('output/beforeProcessing')
    plt.savefig(output_dir / 'targetHisto.png', dpi=200)
    plt.close()
