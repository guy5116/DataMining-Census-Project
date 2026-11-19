import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

#Builds and returns a KNN model
#Splits the data into features and targets
#Splits into trainging and testing sets
#Scales the features for distance calculations
#Iniliatizes and fits the KNN classifier
def make_KNN_model(df):
    #Splits the data into features and targets
    X = df.drop(columns=["income"])
    y = df["income"]

    # Splits into trainging and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Scales the features for distance calculations
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Iniliatizes and fits the KNN classifier
    model = KNeighborsClassifier(
        n_neighbors=5,
    )

    model.fit(X_train_scaled, y_train)
    return model, X_test_scaled, y_test

# Builds and returns an XGBoost model using a train-test split
# Splits the data into features and target
# Splits into training and testing sets
# Initializes and fits the XGBoost classifier
def make_XGB_model(df):
    # Splits the data into features and target
    X = df.drop(columns=["income"])
    y = df["income"]

    # Splits into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initializes and fits the XGBoost classifier
    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
#       sub_sample=1,
        min_child_weight=2,
#       reg_alpha=1.0,
        reg_lambda=1.0,
        scale_pos_weight=1,
#       tree_methods="hist",
        enable_categorical=True,
        random_state=42
    )

    model.fit(X_train, y_train)    

    return model, X_test, y_test

# Builds and returns a Decision Tree model using a train-test split
# Splits the data into features and target
# Splits into training and testing sets
# Initializes and fits the Decision Tree classifier
def make_DTC_model(df):
    # Splits the data into features and target
    X = df.drop(columns=["income"])
    y = df["income"]

    # Splits into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initializes and fits the Decision Tree classifier
    model = DecisionTreeClassifier(
    )

    model.fit(X_train, y_train)    

    return model, X_test, y_test

# Builds and returns an Extra Trees model using a train-test split
# Splits the data into features and target
# Splits into training and testing sets
# Initializes and fits the Extra Trees classifier
def make_ETC_model(df):
    # Splits the data into features and target
    X = df.drop(columns=["income"])
    y = df["income"]

    # Splits into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initializes and fits the Extra Trees classifier
    model = ExtraTreesClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )

    model.fit(X_train, y_train)    

    return model, X_test, y_test

# Builds and returns a Random Forest model using a train-test split
# Splits the data into features and target
# Splits into training and testing sets
# Initializes and fits the Random Forest classifier
def make_RFC_model(df):
    # Splits the data into features and target
    X = df.drop(columns=["income"])
    y = df["income"]

    # Splits into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Initializes and fits the Random Forest classifier
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model, X_test, y_test

# Builds and returns an SVM model with feature scaling and probability calibration
# Splits the data into features and target
# Splits into training and testing sets
# Scales the features for SVM optimization
# Initializes and fits the calibrated SVM classifier
def make_SVM_model(df):
    # Splits the data into features and target
    X = df.drop(columns=["income"])
    y = df["income"]

    # Splits into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scaler lives inside the pipeline so K-Fold re-scales each fold
    # (without it, cross_validate fits the RBF SVC on raw features and it effectively hangs)
    model = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test

# Ranks and prints the top feature importances from a model
# Extracts and sorts feature importances
def print_feature_importances(model, X, top_n=10):
    #Extracts and sorts feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(feature_importance_df.head(top_n).to_string(index=False))

# Evaluates model predictions and prints accuracy, F1, ROC, and classification report
# Generates predictions and calculates evaluation metrics
# Prints out formatted results
def model_testing(model, X_test, y_test):
    # Generates predictions and calculates evaluation metrics
    # print(type(model))
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="binary")
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Prints out formatted results
    print(f"Model Trained Success")
    print(f"Accuracy:   {accuracy*100:.2f}%")
    print(f"F1 Score:   {f1*100:.2f}%")
    print(f"Roc:   {roc*100:.2f}%")
    print(classification_report(y_test, predictions))

# Evaluates any scikit-learn model using Stratified K-Fold Cross-Validation
# Evaluates any scikit-learn compatible model using Stratified K-Fold Cross-Validation
# Initialize Stratified K-Fold
# Define metrics to track
# Run cross-validation
# Calculate mean percentages
# returns the cross-validation results
def evaluate_with_kfold(model, X, y, n_splits=10):

    # Evaluates any scikit-learn compatible model using Stratified K-Fold Cross-Validation
    # Initialize Stratified K-Fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define metrics to track
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Run cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    
    # Calculate mean percentages
    mean_acc = cv_results['test_accuracy'].mean() * 100
    mean_f1 = cv_results['test_f1'].mean() * 100
    mean_roc = cv_results['test_roc_auc'].mean() * 100
    
    print(f"--- {n_splits}-Fold Cross-Validation Results ---")
    print(f"Accuracy:  {mean_acc:.2f}%")
    print(f"F1 Score:  {mean_f1:.2f}%")
    print(f"ROC-AUC:   {mean_roc:.2f}%")
    print("-" * 40)
    
    return cv_results
    
