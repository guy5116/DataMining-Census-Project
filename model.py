import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

def make_KNN_model(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = KNeighborsClassifier(
        n_neighbors=5,
    )

    model.fit(X_train_scaled, y_train)
    return model, X_test_scaled, y_test

def make_XGB_model(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

#   
    
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

def make_DTC_model(df):
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

#   
    
    model = DecisionTreeClassifier(
    )

    model.fit(X_train, y_train)    

    return model, X_test, y_test

def make_ETC_model(df):

    #Builds and returns an Extra Trees Classifier model along with features (X_test) and target (y_test) using a standard train-test split, matching the pattern of the other models.
    X = df.drop(columns=["income"])
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = ExtraTreesClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1
    )

    model.fit(X_train, y_train)    

    return model, X_test, y_test

def print_feature_importances(model, X, top_n=10):
    """
    Ranks and prints how much each attribute affects the class label using Extra Trees feature importances.
    """
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(feature_importance_df.head(top_n).to_string(index=False))

def model_testing(model, X_test, y_test):
#    print(type(model))

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions, average="binary")
    roc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Model Trained Success")
    print(f"Accuracy:   {accuracy*100:.2f}%")
    print(f"F1 Score:   {f1*100:.2f}%")
    print(f"Roc:   {roc*100:.2f}%")
    print(classification_report(y_test, predictions))

def evaluate_with_kfold(model, X, y, n_splits=10):

    #Evaluates any scikit-learn compatible model using Stratified K-Fold Cross-Validation
    # Initialize Stratified K-Fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define metrics to track
    scoring = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Run cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
    
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
    
