from pathlib import Path

from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd

from preprocessing import loadData
from info import compute_info_gain

# Creates a chart that shows the score for every feature by how much it changes
# the entropy of income (Based entropy is 0.81 bits)
# Then creates the bar chart (it only uses clean rows)
# We found the strongest predictors are:
# Relationship (0.166), marital-status (0.157), and education (0.093)
# We found the weakest to be:
# race (0.001), native-country (0.008), and workclass (0.009)
def infoGainChart():
    df = loadData().dropna()
    ig = compute_info_gain(df)

    ig = ig.sort_values('info_gain')            # ascending, so barh puts the biggest on top
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

# Creates a Tree Plot using a depth-3 decision tree with criterion='entropy'
# so every split is highest entropy choice at that point, and draws it
# We found that the first split is marita-status_Married-civ-spouse 
# (Which is expected from the info gain chart)
# 93% of unmarried people are <=50 unless they have over $7,074 capital gains
# Married people are clase to 50/50 and education-num and capital gains then
# seperate the >50k earners
def entropyTreePlot(depth=3):
    df = loadData().dropna()
    X = pd.get_dummies(df.drop(columns=['income']))
    y = df['income'].map({False: '<=50K', True: '>50K'})

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
