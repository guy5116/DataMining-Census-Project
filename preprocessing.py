import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def loadData():
    #Load form adult.csv
    df = pd.read_csv('adult.csv').replace("?", np.nan)
    
    #Prep it for modeling
    #Set over 50k to true and under to false
    df.drop_duplicates
    mapping = {'<=50K': True, '>50K': False}
    df["income"] = df["income"].map(mapping)
    
    #print(df.columns)
    
    #Prepare categorical columns to integers
    #nominal_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    #df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    
    return df

#Split away dirty and clean to a test model with only clean data
def split_data(df):
    dirty_mask = df.isna().any(axis=1)

    clean_df = df[~dirty_mask]
    dirty_df = df[dirty_mask]
    #clean_df.to_csv('c_test.csv')
    #dirty_df.to_csv('d_test.csv')
    return clean_df, dirty_df

def encode_dummies(df):
    #Prepare categorical columns to integers
    nominal_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    #df_encoded.to_csv('output/test.csv')
    return df_encoded


def imputer_cleaning(df, model):
    imputer = IterativeImputer(random_state=42)
    clean_df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return clean_df

def merge_data(df1, df2):
    missing = df1.columns.difference(df2.columns)
    df2 = df2.assign(**{col: False for col in missing})
    full_df = pd.concat([df1, df2], ignore_index=True)
    full_df.to_csv('output/full_adult.csv')
    print(f"DF1 Shape: {df1.shape}\nDF2 Shape: {df2.shape}\nFull DF Shape: {full_df.shape}")
    return full_df
