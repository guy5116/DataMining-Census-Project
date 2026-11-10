import pandas as pd

from graphs import *
from model import *
from preprocessing import *


def create_final_csv(full_df_encoded):
    dummy_cols = [
        col for col in full_df_encoded.columns if "_" in col
    ]  # or filter by your known prefix list
    non_dummy_cols = [c for c in full_df_encoded.columns if c not in dummy_cols]
    decoded = pd.from_dummies(
        full_df_encoded[dummy_cols],
        sep="_",
        default_category={
            "workclass": "Federal-gov",
            "education": "10th",
            "marital-status": "Divorced",
            "occupation": "Adm-clerical",
            "relationship": "Husband",
            "race": "Amer-Indian-Eskimo",
            "sex": "Female",
            "native-country": "Cambodia",
        },
    )
    fixed_full_df = pd.concat([full_df_encoded[non_dummy_cols], decoded], axis=1)
    fixed_full_df.to_csv("output/final_adult.csv")
    histo(fixed_full_df)
    print(f"!!!!!!!FINAL ADULT CSV CREATED!!!!!!!")


def workflow():
    models = {}


def XGB_workflow():
    print("####################-Starting XGB Workflow-####################\n")
    # Loads the data from csv
    df = loadData()

    # Splits already clean with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)

    # Makes a model with only clean data
    c_model, X, y = make_XGB_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df_encoded, c_model)

    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, dirty_df_encoded)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_XGB_model(full_df_encoded)
    full_df_encoded.to_csv("output/full_adult_encoded_XBG.csv")
    model_testing(full_model, fX, fy)

    create_final_csv(full_df_encoded)

    print("####################-Ending XGB Workflow-####################\n\n")


def KNN_workflow():
    print("####################-Starting KNN Workflow-####################\n")
    # Loads the data from csv
    df = loadData()

    # Splits already clean with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)

    # Makes a model with only clean data
    c_model, X, y = make_KNN_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df_encoded, c_model)

    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, dirty_df_encoded)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_KNN_model(full_df_encoded)
    full_df_encoded.to_csv("output/full_adult_encoded_KNN.csv")
    model_testing(full_model, fX, fy)
    print("####################-Ending KNN Workflow-####################\n\n")


def DTC_workflow():
    print("####################-Starting DTC Workflow-####################\n")

    # Loads the data from csv
    df = loadData()

    # Splits already clean with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)

    # Makes a model with only clean data
    c_model, X, y = make_DTC_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df_encoded, c_model)

    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, dirty_df_encoded)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_DTC_model(full_df_encoded)
    full_df_encoded.to_csv("output/full_adult_encoded_DTC.csv")
    model_testing(full_model, fX, fy)
    print("####################-Ending DTC Workflow-####################\n\n")

def ETC_workflow():
    print("####################-Starting ETC Workflow-####################\n")
    # Loads the data from csv
    df = loadData()

    # Splits clean with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)

    # Makes a model with only clean data
    c_model, X, y = make_ETC_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df_encoded, c_model)

    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, dirty_df_encoded)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_ETC_model(full_df_encoded)
    full_df_encoded.to_csv('output/full_adult_encoded_ETC.csv')
    model_testing(full_model, fX, fy)
    
    # Extract and display feature importances
    print("\n--- Top Feature Importances (Extra Trees) ---")
    print_feature_importances(full_model, fX, top_n=10)
    
    print("####################-Ending ETC Workflow-####################\n\n")
    
def run_xgboost_kfold_workflow():
    print("####################-Starting XGB K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)
    
    print("\n--- Evaluating XGB with CLEAN Data Only ---")
    c_model, _, _ = make_XGB_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df_encoded, c_model)
    full_df = merge_data(clean_df_encoded, dirty_df_encoded)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating XGB with FULL (Imputed) Dataset ---")
    full_model, fX, _ = make_XGB_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("####################-Ending XGB K-Fold Workflow-####################\n")

def run_knn_kfold_workflow():
    print("####################-Starting KNN K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)
    
    print("\n--- Evaluating KNN with CLEAN Data Only ---")
    c_model, _, _ = make_KNN_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df_encoded, c_model)
    full_df = merge_data(clean_df_encoded, dirty_df_encoded)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating KNN with FULL (Imputed) Dataset ---")
    full_model, _, _ = make_KNN_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("####################-Ending KNN K-Fold Workflow-####################\n")

def run_dtc_kfold_workflow():
    print("####################-Starting DTC K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)
    
    print("\n--- Evaluating DTC with CLEAN Data Only ---")
    c_model, _, _ = make_DTC_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df_encoded, c_model)
    full_df = merge_data(clean_df_encoded, dirty_df_encoded)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating DTC with FULL (Imputed) Dataset ---")
    full_model, _, _ = make_DTC_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("####################-Ending DTC K-Fold Workflow-####################\n")

def run_etc_kfold_workflow():
    print("####################-Starting ETC K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)
    
    print("\n--- Evaluating ETC with CLEAN Data Only ---")
    c_model, _, _ = make_ETC_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df_encoded, c_model)
    full_df = merge_data(clean_df_encoded, dirty_df_encoded)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating ETC with FULL (Imputed) Dataset ---")
    full_model, fX, _ = make_ETC_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("\n--- Top Feature Importances (Extra Trees - K-Fold Dataset) ---")
    print_feature_importances(full_model, fX, top_n=10)
    
    print("####################-Ending ETC K-Fold Workflow-####################\n")


