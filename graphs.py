# import pandas as pd
# import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from preprocessing import loadData
from info import *

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

def capitalBoxPlots():
    df = loadData()
    cols = ['capital-gains', 'capital-loss']

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col in zip(axes, cols):
        groups = [df.loc[df['income'] == label, col] for label in (True, False)]
        ax.boxplot(groups, tick_labels=['<=50K', '>50K'])
        ax.set_yscale('symlog')
        zero_pct = (df[col] == 0).mean() * 100
        ax.set_title(f"{col}\n({zero_pct:.1f}% of rows are 0)")
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Capital gains and losses by income (symlog scale)')
    plt.tight_layout()
    output_dir = Path('output/beforeProcessing')
    plt.savefig(output_dir / 'capital_boxplots.png', dpi=200)
    plt.close

def histo(df):
    plt.hist(df["income"].astype(int), bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1], ["True", "False"])
    plt.show()



def infoGainChart():
    df = loadData().dropna()
    ig = compute_info_gain(df)

    ig = ig.sort_values('info_gain')
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(ig['feature'], ig['info_gain'], color='#4C72B0')
    for bar, gain in zip(bars, ig['info_gain']):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{gain:.3f}", va='center', fontsize=9)

    ax.set_xlabel('Information gain (bits)')
    ax.set_title('Information gain of each feature for predicting income')
    ax.set_xlim(0, ig['info_gain'].max() * 1.15)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    output_dir = Path('output/InfoGain')
    plt.savefig(output_dir / 'info_gain.png', dpi=200)
    plt.close()

def entropyTreePlot(depth=3):
    df = loadData().dropna()
    X = pd.get_dummies(df.drop(columns=['income']))
    y = df['income'].map({True: '<=50K', False: '>50K'})

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=depth, random_state=42)
    tree.fit(X, y)

    fig, ax = plt.subplots(figsize=(20,9))
    plot_tree(tree, feature_names=X.columns, class_names=tree.classes_,
              filled=True, rounded=True, impurity=True, proportion=True,
              fontsize=9, ax=ax)
    ax.set_title(f"Decision tree (entropy, depth {depth} on clean rows)")

    plt.tight_layout()
    output_dir = Path('output/InfoGain')
    plt.savefig(output_dir / 'entropy_tree.png', dpi=150)
    plt.close()



if __name__ == "__main__":
    infoGainChart()
    entropyTreePlot()




