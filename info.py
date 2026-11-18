from preprocessing import loadData
import pandas as pd
import numpy as np

# Entropy of a column of class labels.
# value_counts(normalize=True) gives the share of each class
# then we apply H = -sum(p * log2 p).
# For out income column this is about 0.81 bits
def entropy(labels):
    """H(Y) = -sum(p * log2 p) over the classes in labels. """
    p = labels.value_counts(normalize=True)
    return -(p * np.log2(p)).sum()

# Information gain for a categorical feature
# Each feature category becomes its own child group
def info_gain_categorical(df, feature, target='income'):
    parent = entropy(df[target])
    weight = df[feature].value_counts(normalize=True)
    children = df.groupby(feature)[target].apply(entropy)
    return parent - (weight * children).sum()

# Information gain for a numeric feature (age, hours-per, week, ...)
# A number can't be split once per value, so we do what a decision tree does:
#       try ever threshold 'feature <= t' and keep the one with the highest gain
# Returns (best gain, best threshold)
def info_gain_numeric(df, feature, target='income'):
    # Sort by the feature so every possible cut is just 'first k rows vs the rest'
    data = df[[feature, target]].sort_values(feature)
    x = data[feature].to_numpy()
    y = data[target].to_numpy().astype(int)         # True/False -> 1/0
    n, total_pos = len(y), y.sum()
    
    # Running total let us score every threshold in one pass, no loop:
    #       pos_left[k] = number of 1s in the first k+1 rows, n_left[k] = k+1 rows
    pos_left = np.cumsum(y)[:-1]
    n_left = np.arange(1, n)

    # Only cut between two different values (can't split age 30 from age 30)
    valid = x[:-1] != x[1:]
    pos_left, n_left = pos_left[valid], n_left[valid]
    n_right = n - n_left
    pos_right = total_pos - pos_left
    
    # Binary entropy for many groups at once: p = share of 1s in each group
    def H(pos, cnt):
        p = pos/cnt
        # p = 0 or 1 gives log2(0); numpy wanrs the reurns nan, which we
        # turn into 0 below
        with np.errstate(divide='ignore', invalid='ignore'):
            h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        return np.nan_to_num(h)
    
    # Weighted entropy of the left and right sides for every threshold
    child = (n_left * H(pos_left, n_left) + n_right * H(pos_right, n_right)) / n
    gains = entropy(data[target]) - child
    best = gains.argmax()               # Threshold with the largest gain
    threshold = x[:-1][valid][best]
    return gains[best], threshold

# Returns the right IG function on every column and returns a table:
#       feature | info_gain | split
# Sorted from most to least informative
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

