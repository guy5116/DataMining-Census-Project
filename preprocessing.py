import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Loads the data from the adult.csv file
# replaces '?' with np.nan
# Drops the duplicates
# Maps the target so under or equal to 50k is False where over is True
# Returns the data frame
def loadData():
    # Load form adult.csv
    df = pd.read_csv("adult.csv").replace("?", np.nan)

    # Prep it for modeling
    # Set over 50k to true and under to false
    df.drop_duplicates()
    mapping = {"<=50K": False, ">50K": True}
    df["income"] = df["income"].map(mapping)

    # print(df.columns)

    # Prepare categorical columns to integers
    # nominal_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    # df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    return df


# Splits the data frame from already clean and rows with missing values
# Returns the clean and dirty data frames
def split_data(df):
    dirty_mask = df.isna().any(axis=1)

    clean_df = df[~dirty_mask]
    dirty_df = df[dirty_mask]
    # clean_df.to_csv('c_test.csv')
    # dirty_df.to_csv('d_test.csv')
    return clean_df, dirty_df

# Converts the categorical features into multiple boolean features
# Returns the new data frame
def encode_dummies(df):
    # Prepare categorical columns to integers
    nominal_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    # df_encoded.to_csv('output/test.csv')
    return df_encoded

# Imputes the missing values in the dirty data frame
# Uses the clean encoded data frame to line up the columns and help train the imputer
# returns the clean data frame
def imputer_cleaning(dirty_df, clean_df_encoded):
    # Encode and match the clean columns (categories missing here become False)
    encoded = encode_dummies(dirty_df).reindex(columns=clean_df_encoded.columns, fill_value=False).astype(float)

    # get_dummies drops NaN, so put the NaN back into that feature's dummy columns
    missing_cols = dirty_df.columns[dirty_df.isna().any()]
    for col in missing_cols:
        encoded.loc[dirty_df[col].isna().values, encoded.columns.str.startswith(col + "_")] = np.nan

    # Fit on clean + dirty rows so the imputer learns from the clean data too
    imputer = IterativeImputer(random_state=42)
    imputer.fit(pd.concat([clean_df_encoded.astype(float), encoded]))
    clean_df = pd.DataFrame(imputer.transform(encoded), columns=encoded.columns)

    # Turn the imputed scores back into one category per row
    # (drop_first removed one category, its score is 1 - sum of the others)
    for col in missing_cols:
        group = clean_df.columns[clean_df.columns.str.startswith(col + "_")]
        scores = clean_df[group]
        best = scores.idxmax(axis=1)
        is_base = scores.max(axis=1) < 1 - scores.sum(axis=1)
        clean_df[group] = False
        for c in group:
            clean_df.loc[(best == c) & ~is_base, c] = True

    # Back to the original types (bool dummies/income, int numbers)
    return clean_df.astype(clean_df_encoded.dtypes)

# Takes two data frames and merges them
# Returns the merged data frame
def merge_data(df1, df2):
    missing = df1.columns.difference(df2.columns)
    df2 = df2.assign(**{col: False for col in missing})
    full_df = pd.concat([df1, df2], ignore_index=True)
    full_df.to_csv("output/full_adult.csv")
    print(
        f"DF1 Shape: {df1.shape}\nDF2 Shape: {df2.shape}\nFull DF Shape: {full_df.shape}"
    )
    return full_df
