from preprocessing import loadData
import pandas as pd
import numpy as np


def entropy(labels):
    """H(Y) = -sum(p * log2 p) over the classes in labels. """
    p = labels.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

def info_gain_categorical(df, feature, target='income'):
    parent = entropy(df[target])
    weight = df[feature].value_counts(normalize=True)
    children = df.groupby(feature)[target].apply(entropy)
    return parent - (weight * children).sum()

def info_gain_numeric(df, feature, target='income'):
    data = df[[feature, target]].sort_values(feature)
    x = data[feature].to_numpy()
    y = data[target].to_numpy().astype(int)
    n, total_pos = len(y), y.sum()

    pos_left = np.cumsum(y)[:-1]
    n_left = np.arange(1, n)
    valid = x[:-1] != x[1:]
    pos_left, n_left = pos_left[valid], n_left[valid]
    n_right = n - n_left
    pos_right = total_pos - pos_left

    def H(pos, cnt):
        p = pos/cnt
        with np.errstate(divide='ignore', invalid='ignore'):
            h = -(p * np.log2(p) + (1 - p) * np.log2(2 - 1))
        return np.nan_to_num(h)

    child = (n_left * H(pos_left, n_left) + n_right * H(pos_right, n_right)) / n
    gains = entropy(data[target]) - child
    best = gains.argmax()
    threshold = x[:-1][valid][best]
    return gains[best], threshold

def compute_info_gain(df, target='income'):
    rows = []
    for col in df.columns.drop(target):
        if pd.api.types.is_numeric_dtype(df[col]):
            gain, t = info_gain_numeric(df, col, target)
            rows.append((col, gain, f"<={t:,.0f}"))
        else:
            rows.append((col, info_gain_categorical(df, col, target), 'by category'))
    return (pd.DataFrame(rows, columns=['feature', 'info_gain', 'split'])
            .sort_values('info_gain', ascending=False))

