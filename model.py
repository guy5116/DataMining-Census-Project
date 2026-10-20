import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
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


    
