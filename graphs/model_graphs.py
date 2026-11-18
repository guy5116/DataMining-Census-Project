from pathlib import Path

from sklearn.metrics import ConfusionMatrixDisplay, auc, average_precision_score, confusion_matrix, roc_curve, precision_recall_curve, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

from model import make_XGB_model, make_KNN_model, make_DTC_model, make_ETC_model
from preprocessing import loadData, split_data, encode_dummies, imputer_cleaning, merge_data

OUTPUT_DIR = Path('output/models')


MODEL_BUILDERS = {
        'XGB' : make_XGB_model,
        'KNN' : make_KNN_model,
        'DTC' : make_DTC_model,
        'ETC' : make_ETC_model,
        }

def build_full_data():
    df = loadData()
    clean_df, dirty_df = split_data(df)
    clean_df_encoded = encode_dummies(clean_df)
    c_dirty_df = imputer_cleaning(dirty_df, clean_df_encoded)
    full_df = merge_data(clean_df_encoded, c_dirty_df)
    return encode_dummies(full_df)

def train_models(df_encoded):
    models = {}
    for name, builder in MODEL_BUILDERS.items():
        print(f"Training {name}...")
        models[name] = builder(df_encoded)
    return models

def confusionMatices(models):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    for ax, (name, (model, X_test, y_test)) in zip(axes.flat, models.items()):
        y_true = np.asarray(y_test).astype(int)
        y_pred = np.asarray(model.predict(X_test)).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        ConfusionMatrixDisplay(cm, display_labels=['<=50K', '>50K']).plot(
                ax=ax, cmap='Blues', colorbar=False, values_format=','
                )

        recall = cm[1, 1] / cm[1].sum()
        accuracy = np.trace(cm) / cm.sum()
        ax.set_title(f"{name} (accuracy {accuracy:.1%})\ncaught {recall:.1%} of >50K earners")

    fig.suptitle('Confusion matrices on the test set (full imputed data)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=200)
    plt.close()

def rocCurve(models):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))

    for name, (model, X_test, y_test) in models.items():
        y_true = np.asarray(y_test).astype(int)
        scores = model.predict_proba(X_test)[:,1]

        fpr, tpr, _ = roc_curve(y_true, scores)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle='--', color='grey', label='Random guess (AUC = 0.500)')

    ax.set_xlabel('False positive rate (<=50K wrongly flagged as >50K)')
    ax.set_ylabel('True positive rate (>50K correctly caught)')
    ax.set_title('ROC curves on the test set (full imputed data)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect('equal')
    ax.legend(loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=200)
    plt.close()

def precisionRecallCurves(models):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7,7))

    for name, (model, X_test, y_test) in models.items():
        y_true = np.asarray(y_test).astype(int)
        scores = model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        line, = ax.plot(recall, precision, linewidth=2, label=f"{name} (AP = {ap:.3f})")

        i = np.searchsorted(thresholds, 0.5, side='right')
        if i < len(thresholds):
            ax.plot(recall[i], precision[i], 'o', color=line.get_color(), markersize=8)

    base_rate = np.asarray(y_test).astype(int).mean()
    ax.axhline(base_rate, linestyle='--', color='gray',
              label=f"Random guess (precision = {base_rate:.2f})")

    ax.set_xlabel('Recall (share of >50K earners caught)')
    ax.set_ylabel('Precision (share of >50K predictions that are right)')
    ax.set_title('Precision-recall curves on the test set (dots = 0.5 cutoff)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect('equal')
    ax.legend(loc='lower left')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'precision_recall_curves.png', dpi=200)
    plt.close()

def kfoldBoxPlots(models, df_encoded, n_splits=10):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X = df_encoded.drop(columns=['income'])
    y = df_encoded['income'].astype(int)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    f1_scores, auc_scores = {}, {}
    for name, (model, _, _) in models.items():
        if isinstance(model, KNeighborsClassifier):
            model = make_pipeline(StandardScaler(), model)

        result = cross_validate(model, X, y, cv=cv, scoring=['f1', 'roc_auc'])
        f1_scores[name] = result['test_f1']
        auc_scores[name] = result['test_roc_auc']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, scores, title in zip(axes, (f1_scores, auc_scores), ('F1 score (>50K class)', 'ROC-AUC')):
        names = list(scores)
        ax.boxplot([scores[n] for n in names], tick_labels=names)

        rng = np.random.default_rng(42)
        for i, n in enumerate(names, start=1):
            ax.scatter(i + rng.uniform(-0.08, 0.08, len(scores[n])), scores[n],
                       s=14, alpha=0.6, color='#4C72B0', zorder=3)

        ax.set_title(title)
        ax.set_ylabel('Score per fold')
        ax.grid(axis='y', alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle(f"{n_splits}-fold cross-validation scores per model (full imputed data)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'kfold_boxplots.png', dpi=200)
    plt.close()

def score_model(model, X_test, y_test):
    y_true = np.asarray(y_test, y_test)
    y_pred = np.asarray(model.predict(X_test)).astype(int)
    scores = model.predict_proba(X_test)[:,1]
    return f1_score(y_true, y_pred), roc_auc_score(y_true, scores)

def cleanVsImputed(full_models):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    clean_df, _ = split_data(loadData())
    clean_models = train_models(encode_dummies(clean_df))

    names = list(full_models)
    clean_scores = [score_model(*clean_models[n]) for n in names]
    full_scores = [score_model(*full_models[n]) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(names))
    width = 0.38

    for ax, metric_index, title in zip(axes, (0, 1), ('F1 score (>50K)', 'ROC-AUC')):
        clean_vals = [s[metric_index] for s in clean_scores]
        full_vals = [s[metric_index] for s in full_scores]

        b1 = ax.bar(x - width / 2, clean_vals, width, label='Clean rows only', color='#4C72B0')
        b2 = ax.bar(x + width / 2, full_vals, width, label='With imputed rows', color='#DD8452')
        ax.bar_label(b1, fmt='%.3f', fontsize=8, padding=2)
        ax.bar_label(b2, fmt='%.3f', fontsize=8, padding=2)

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.spines[['top', 'right']].set_visible(False)

    axes[0].legend(loc='lower left')
    fig.suptitle('Did imputing the dirty rows help? Clean-only vs Imputed training data', fontsize=13)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'clean_vs_imputed.png', dpi=200)
    plt.close()
