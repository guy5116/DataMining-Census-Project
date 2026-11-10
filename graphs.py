# import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt


def histo(df):
    plt.hist(df["income"].astype(int), bins=[-0.5, 0.5, 1.5], rwidth=0.8)
    plt.xticks([0, 1], ["True", "False"])
    plt.show()
