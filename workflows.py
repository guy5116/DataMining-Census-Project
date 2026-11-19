import pandas as pd

from model import (make_XGB_model, make_KNN_model, make_DTC_model, make_ETC_model,
                   model_testing, print_feature_importances, evaluate_with_kfold)
from preprocessing import loadData, split_data, encode_dummies, imputer_cleaning, merge_data

# Rencodes the Data Frame with default categories
# Switches it from multiple boolean columns to a categorical columns
# Also creates a histogram from this CSV
def create_final_csv(full_df_encoded):
    dummy_cols = [
        col for col in full_df_encoded.columns if "_" in col
    ]
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
    print(f"!!!!!!!FINAL ADULT CSV CREATED!!!!!!!")

# Runs the XGB workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Makes the XGB model using the clean data frame
# Tests the model
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a XGB model with the Merged data frame
# Outputs the Merged data frame to 'outputs/XGB/' as 'full_adult_encoded_XGB.csv'
# Tests the model
# Creates the final CSV
def XGB_workflow():
    print("####################-Starting XGB Workflow-####################\n")
    # Loads the data from csv
    df = loadData()

    # Splits already clean with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)

    # Makes a model with only clean data
    c_model, X, y = make_XGB_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)

    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_XGB_model(full_df_encoded)
    full_df_encoded.to_csv("output/XGB/full_adult_encoded_XBG.csv")
    model_testing(full_model, fX, fy)

    create_final_csv(full_df_encoded)

    print("####################-Ending XGB Workflow-####################\n\n")

# Runs the KNN workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Makes the KNN model using the clean data frame
# Tests the model
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a KNN model with the Merged data frame
# Outputs the Merged data frame to 'outputs/KNN/' as 'full_adult_encoded_KNN.csv'
# Tests the model
def KNN_workflow():
    print("####################-Starting KNN Workflow-####################\n")
    # Loads the data from csv
    df = loadData()

    # Splits already clean with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)

    # Makes a model with only clean data
    c_model, X, y = make_KNN_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)

    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_KNN_model(full_df_encoded)
    full_df_encoded.to_csv("output/KNN/full_adult_encoded_KNN.csv")
    model_testing(full_model, fX, fy)
    print("####################-Ending KNN Workflow-####################\n\n")

# Runs the DTC workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Makes the DTC model using the clean data frame
# Tests the model
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a DTC model with the Merged data frame
# Outputs the Merged data frame to 'outputs/DTC/' as 'full_adult_encoded_DTC.csv'
# Tests the model
def DTC_workflow():
    print("####################-Starting DTC Workflow-####################\n")

    # Loads the data from csv
    df = loadData()

    # Splits already clean with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)

    # Makes a model with only clean data
    c_model, X, y = make_DTC_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)

    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_DTC_model(full_df_encoded)
    full_df_encoded.to_csv("output/DTC/full_adult_encoded_DTC.csv")
    model_testing(full_model, fX, fy)
    print("####################-Ending DTC Workflow-####################\n\n")

# Runs the ETC workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Makes the ETC model using the clean data frame
# Tests the model
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a ETC model with the Merged data frame
# Outputs the Merged data frame to 'outputs/ETC/' as 'full_adult_encoded_ETC.csv'
# Tests the model
def ETC_workflow():
    print("####################-Starting ETC Workflow-####################\n")
    # Loads the data from csv
    df = loadData()

    # Splits clean with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)

    # Makes a model with only clean data
    c_model, X, y = make_ETC_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)

    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_ETC_model(full_df_encoded)
    full_df_encoded.to_csv('output/ETC/full_adult_encoded_ETC.csv')
    model_testing(full_model, fX, fy)
    
    # Extract and display feature importances
    print("\n--- Top Feature Importances (Extra Trees) ---")
    print_feature_importances(full_model, fX, top_n=10)
    
    print("####################-Ending ETC Workflow-####################\n\n")

# Runs the RFC workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Makes the RFC model using the clean data frame
# Tests the model
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a RFC model with the Merged data frame
# Outputs the Merged data frame to 'outputs/RFC/' as 'full_adult_encoded_RFC.csv'
# Tests the model
def RFC_workflow():
    print("####################-Starting RFC Workflow-####################\n")
    # Loads the data
    df = loadData()

    # Splits the already clean data with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)

    #Makes a model with only clean data
    c_model, X, y = make_RFC_model(clean_df_encoded)
    model_testing(c_model, X, y)

    #Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df_encoded, clean_df_encoded)
    
    #Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)

    #Makes a model with the merged data
    full_model, fX, fy = make_RFC_model(full_df_encoded)
    full_df_encoded.to_csv('output/RFC/full_adult_encoded_RFC.csv', errors="ignore")
    model_testing(full_model, fX, fy)

    #Extract and display feature importances
    print("\n--- Top Feature Importances (Random Forest) ---")
    print_feature_importances(full_model, fX, top_n=10)
    print("####################-Ending RFC Workflow-####################\n\n")

# Runs the SVM workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Makes the SVM model using the clean data frame
# Tests the model
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a SVM model with the Merged data frame
# Outputs the Merged data frame to 'outputs/SVM/' as 'full_adult_encoded_SVM.csv'
# Tests the model
def SVM_workflow():
    print("####################-Starting SVM Workflow-####################\n")

    # Loads the data
    df = loadData()

    # Splits the already clean data with noisy data
    clean_df, dirty_df = split_data(df)

    # Encodes the data
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)

    # Makes a model with only clean data
    c_model, X, y = make_SVM_model(clean_df_encoded)
    model_testing(c_model, X, y)

    # Cleans the data using our model
    c_dirty_df = imputer_cleaning(dirty_df_encoded, clean_df_encoded)
    
    # Merges the now clean data with the already clean data
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)

    # Makes a model with the merged data
    full_model, fX, fy = make_SVM_model(full_df_encoded)
    full_df_encoded.to_csv('output/SVM/full_adult_encoded_SVM.csv', errors="ignore")
    model_testing(full_model, fX, fy)
    print("####################-Ending SVM Workflow-####################\n\n")
    
# Runs the XGB K-FOLD workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Encodes the categorical columns in borth clean and dirty data frame
# Makes the XGB model using the clean data frame
# Evaluates the model with K-FOLD using only the clean model and clean data frame
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a XGB model with the Merged data frame
# Evaluates the model with K-FOLD using the Full model and Merged data frame
def run_xgboost_kfold_workflow():
    print("####################-Starting XGB K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    
    print("\n--- Evaluating XGB with CLEAN Data Only ---")
    c_model, _, _ = make_XGB_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating XGB with FULL (Imputed) Dataset ---")
    full_model, fX, _ = make_XGB_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("####################-Ending XGB K-Fold Workflow-####################\n")

# Runs the KNN K-FOLD workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Encodes the categorical columns in borth clean and dirty data frame
# Makes the KNN model using the clean data frame
# Evaluates the model with K-FOLD using only the clean model and clean data frame
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a KNN model with the Merged data frame
# Evaluates the model with K-FOLD using the Full model and Merged data frame
def run_knn_kfold_workflow():
    print("####################-Starting KNN K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    
    print("\n--- Evaluating KNN with CLEAN Data Only ---")
    c_model, _, _ = make_KNN_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating KNN with FULL (Imputed) Dataset ---")
    full_model, _, _ = make_KNN_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("####################-Ending KNN K-Fold Workflow-####################\n")


# Runs the DTC K-FOLD workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Encodes the categorical columns in borth clean and dirty data frame
# Makes the DTC model using the clean data frame
# Evaluates the model with K-FOLD using only the clean model and clean data frame
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a DTC model with the Merged data frame
# Evaluates the model with K-FOLD using the Full model and Merged data frame
def run_dtc_kfold_workflow():
    print("####################-Starting DTC K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    
    print("\n--- Evaluating DTC with CLEAN Data Only ---")
    c_model, _, _ = make_DTC_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating DTC with FULL (Imputed) Dataset ---")
    full_model, _, _ = make_DTC_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("####################-Ending DTC K-Fold Workflow-####################\n")

# Runs the ETC K-FOLD workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Encodes the categorical columns in borth clean and dirty data frame
# Makes the ETC model using the clean data frame
# Evaluates the model with K-FOLD using only the clean model and clean data frame
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a ETC model with the Merged data frame
# Evaluates the model with K-FOLD using the Full model and Merged data frame
def run_etc_kfold_workflow():
    print("####################-Starting ETC K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    
    print("\n--- Evaluating ETC with CLEAN Data Only ---")
    c_model, _, _ = make_ETC_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating ETC with FULL (Imputed) Dataset ---")
    full_model, fX, _ = make_ETC_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("\n--- Top Feature Importances (Extra Trees - K-Fold Dataset) ---")
    print_feature_importances(full_model, fX, top_n=10)
    
    print("####################-Ending ETC K-Fold Workflow-####################\n")

# Runs the RFC K-FOLD workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Encodes the categorical columns in borth clean and dirty data frame
# Makes the RFC model using the clean data frame
# Evaluates the model with K-FOLD using only the clean model and clean data frame
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a RFC model with the Merged data frame
# Evaluates the model with K-FOLD using the Full model and Merged data frame
def run_rfc_kfold_workflow():
    print("####################-Starting RFC K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)
    
    print("\n--- Evaluating RFC with CLEAN Data Only ---")
    c_model, _, _ = make_RFC_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df_encoded, clean_df_encoded)
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating RFC with FULL (Imputed) Dataset ---")
    full_model, fX, _ = make_RFC_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    
    print("\n--- Top Feature Importances (Random Forest - K-Fold Dataset) ---")
    print_feature_importances(full_model, fX, top_n=10)
    print("####################-Ending RFC K-Fold Workflow-####################\n")

# Runs the SVM K-FOLD workflow
# Loads the adult.csv data set
# Splits the data (clean and dirty data frames)
# Encodes the categorical columns in borth clean and dirty data frame
# Makes the SVM model using the clean data frame
# Evaluates the model with K-FOLD using only the clean model and clean data frame
# Uses the imputer to clean the dirty data frame
# Merges the original clean set with the now clean dirty data frame
# Creates a SVM model with the Merged data frame
# Evaluates the model with K-FOLD using the Full model and Merged data frame
def run_svm_kfold_workflow():
    print("####################-Starting SVM K-Fold Workflow-####################")
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    dirty_df_encoded = encode_dummies(dirty_df)
    
    print("\n--- Evaluating SVM with CLEAN Data Only ---")
    c_model, _, _ = make_SVM_model(clean_df_encoded)
    evaluate_with_kfold(c_model, clean_df_encoded.drop(columns=["income"]), clean_df_encoded["income"], n_splits=10)

    c_dirty_df = imputer_cleaning(dirty_df_encoded, clean_df_encoded)
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    full_df_encoded = encode_dummies(full_df)
    
    print("\n--- Evaluating SVM with FULL (Imputed) Dataset ---")
    full_model, _, _ = make_SVM_model(full_df_encoded)
    evaluate_with_kfold(full_model, full_df_encoded.drop(columns=["income"]), full_df_encoded["income"], n_splits=10)
    print("####################-Ending SVM K-Fold Workflow-####################\n")

