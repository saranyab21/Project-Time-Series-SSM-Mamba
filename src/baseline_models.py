"""
Baseline models for PhysioNet - Gait in Parkinson’s Disease.
  - Using nested cross-validation (outer 5-fold, inner 3-fold)
  - Adding PCA-based dimensionality reduction inside the pipeline
  - Using hyperparameter grids (especially for SVM-RBF)
  - Reporting mean ± std metrics across outer folds (Accuracy, Sensitivity, Specificity, AUC)
  - Saving per-fold metrics and aggregated summaries for Left / Right / Combined datasets
  - Saving confusion matrices from aggregated outer-fold predictions (nested CV)

Prerequisites
-------------
Run TSFresh pipeline first so that these files exist in data/processed:

    features_combined_agg.parquet
    subject_overview.csv  (must contain column 'group' with 0=HC, 1=PD)

Outputs
-------
  - reports/baseline_folds.csv
      Per-fold metrics for each Dataset × Model × OuterFold.

  - reports/baseline_result_summary.csv
      Mean ± std metrics for each Dataset × Model.

  - reports/figs/cm_opt_<Dataset>_<Model>.png
      Update after review from Prof. Tomas: Confusion matrices from aggregated outer-fold predictions.

Datasets
--------
  - Left foot  : all columns with prefix 'L_'
  - Right foot : all columns with prefix 'R_'
  - Combined   : all columns (L_ + R_)

Models
------
  - RandomForest
  - SVM (linear kernel)
  - SVM (RBF kernel)

Evaluation
----------
  - Nested Stratified CV:
      Outer loop: 5-fold (generalization estimate)
      Inner loop: 3-fold GridSearchCV (hyperparameter tuning, scoring=ROC AUC)

  - Metrics (per outer-test fold and aggregated):
      * Accuracy
      * Sensitivity (recall for PD, label=1)
      * Specificity (recall for HC, label=0)
      * ROC AUC

Confusion matrices are produced by aggregating out-of-fold predictions from the outer CV loop.
Every subject is predicted exactly once on a held-out fold, consistent with the nested CV results table.
"""

from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

# --------------------------------------------------
# Paths & basic config 
# --------------------------------------------------
DATA_PROC = Path("data/processed")
REPORT_FIGS = Path("reports/figs")
REPORT_FIGS.mkdir(parents=True, exist_ok=True)

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style="whitegrid", context="talk")
plt.rcParams["figure.dpi"] = 130


def clean_index(idx: pd.Index) -> pd.Index:
    """Simple index cleaner: strip whitespace and cast to string."""
    idx = idx.astype(str)
    idx = idx.str.strip()
    return idx


def compute_metrics(y_true, y_pred, y_prob):
    """
    Compute core classification metrics.

    Returns:
        acc  : Accuracy
        sens : Sensitivity (Recall for PD, label=1)
        spec : Specificity (Recall for HC, label=0)
        auc  : ROC AUC
    """
    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)
    auc = roc_auc_score(y_true, y_prob)
    return acc, sens, spec, auc


def load_datasets():
    """
    Load combined TSFresh features and labels, then derive Left/Right/Combined matrices.

    Returns:
        datasets: dict[str, tuple[pd.DataFrame, pd.Series]]
                  { "Left": (X_left, y), "Right": (X_right, y), "Combined": (X_comb, y) }
    """
    print("Loading features_combined_agg.parquet and subject_overview.csv ...")
    X_comb = pd.read_parquet(DATA_PROC / "features_combined_agg.parquet")

    meta = pd.read_csv(DATA_PROC / "subject_overview.csv", index_col="subject_id")
    y_all = meta["group"].astype(int)

    # Clean indices and align
    X_comb.index = clean_index(X_comb.index)
    y_all.index = clean_index(y_all.index)

    common = X_comb.index.intersection(y_all.index)
    if len(common) == 0:
        raise RuntimeError(
            "No overlapping subject IDs between features_combined_agg and subject_overview.csv"
        )

    X_comb = X_comb.loc[common]
    y = y_all.loc[common]

    # Derive Left and Right from Combined using prefixes
    left_cols = [c for c in X_comb.columns if c.startswith("L_")]
    right_cols = [c for c in X_comb.columns if c.startswith("R_")]

    if len(left_cols) == 0 or len(right_cols) == 0:
        print(
            "[WARN] No L_/R_ prefixed columns found. "
            "Check features_combined_agg.parquet column names."
        )

    X_left = X_comb[left_cols].copy()
    X_right = X_comb[right_cols].copy()

    datasets = {
        "Left": (X_left, y),
        "Right": (X_right, y),
        "Combined": (X_comb, y),
    }

    for name, (X, y_vec) in datasets.items():
        print(f"  Dataset {name}: X shape = {X.shape}, y shape = {y_vec.shape}")

    return datasets


def define_models_with_grids():
    """
    Define classifiers and hyperparameter grids including PCA n_components.

    Returns:
        models: dict[str, tuple[estimator, dict]]
    """
    models = {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                "pca__n_components": [30, 60, 120],
                "clf__n_estimators": [200, 400, 800],
                "clf__max_depth": [None, 10, 20, 40],
                "clf__min_samples_split": [2, 5, 10],
            },
        ),
        "SVM_Linear": (
            SVC(kernel="linear", probability=True, random_state=42),
            {
                "pca__n_components": [30, 60, 120],
                "clf__C": [0.01, 0.1, 1, 10, 100],
            },
        ),
        "SVM_RBF": (
            SVC(kernel="rbf", probability=True, random_state=42),
            {
                "pca__n_components": [30, 60, 120],
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__gamma": ["scale", 1e-4, 1e-3, 1e-2],
            },
        ),
    }
    return models


def plot_confusion_matrix(y_true, y_pred, title, out_path):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=450)
    plt.close()
    print(f"Saved confusion matrix to {out_path}")


def main():
    print("=== BASELINE MODELS (Nested CV + PCA) ===")

    # --------------------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------------------
    datasets = load_datasets()

    # --------------------------------------------------------------
    # 2. Define models + hyperparameter grids
    # --------------------------------------------------------------
    models = define_models_with_grids()

    # Outer and inner CV for nested cross-validation
    outer_splits = 5
    inner_splits = 3

    # Collect per-fold metrics, summary stats, and out-of-fold predictions
    rows_folds = []
    summary_stats = []
    # oof_preds[ds_name][model_name] = (y_true_all, y_pred_all)
    oof_preds = defaultdict(dict)

    # --------------------------------------------------------------
    # 3. Nested CV for each dataset and model
    # --------------------------------------------------------------
    for ds_name, (X, y_vec) in datasets.items():
        n_samples, n_features = X.shape
        print("\n" + "=" * 30)
        print(f"DATASET: {ds_name} (n={n_samples}, d={n_features})")
        print("=" * 30)

        if n_samples == 0:
            print(f"[WARN] {ds_name} has 0 samples. Skipping.")
            continue

        if len(np.unique(y_vec)) < 2:
            print(f"[WARN] {ds_name} has only one class in y. Skipping.")
            continue

        # Prepare outer CV for this dataset
        outer_cv = StratifiedKFold(
            n_splits=outer_splits, shuffle=True, random_state=42
        )

        for model_name, (base_clf, param_grid) in models.items():
            print(f"\n--- Model: {model_name} ---")
            # Per-model container for metrics across outer folds
            fold_metrics = []
            # Accumulate out-of-fold predictions for aggregated confusion matrix
            oof_true_list = []
            oof_pred_list = []

            # Nested CV: outer loop for evaluation
            for fold_idx, (train_idx, test_idx) in enumerate(
                outer_cv.split(X, y_vec), start=1
            ):
                print(f"  > Outer fold {fold_idx}/{outer_splits} ...")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y_vec.iloc[train_idx], y_vec.iloc[test_idx]

                # Inner CV for hyperparameter tuning
                inner_cv = StratifiedKFold(
                    n_splits=inner_splits, shuffle=True, random_state=123
                )

                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("pca", PCA()),
                        ("clf", base_clf),
                    ]
                )

                grid = GridSearchCV(
                    estimator=pipe,
                    param_grid=param_grid,
                    cv=inner_cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                )

                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_

                # Evaluate on the held-out outer fold
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]

                acc, sens, spec, auc = compute_metrics(y_test, y_pred, y_prob)
                print(
                    f"    Fold {fold_idx}: "
                    f"Acc={acc:.3f}, Sens={sens:.3f}, Spec={spec:.3f}, AUC={auc:.3f}"
                )

                fold_metrics.append((acc, sens, spec, auc))

                # Accumulate out-of-fold predictions
                oof_true_list.extend(y_test.tolist())
                oof_pred_list.extend(y_pred.tolist())

                rows_folds.append(
                    {
                        "Dataset": ds_name,
                        "Model": model_name,
                        "Fold": fold_idx,
                        "Accuracy": acc,
                        "Sensitivity": sens,
                        "Specificity": spec,
                        "AUC": auc,
                        "BestParams": best_params,
                    }
                )

            # Store aggregated out-of-fold predictions for this dataset × model
            oof_preds[ds_name][model_name] = (
                np.array(oof_true_list),
                np.array(oof_pred_list),
            )

            # Summarize across outer folds for this Dataset × Model
            if not fold_metrics:
                continue

            metrics_arr = np.array(fold_metrics)  # shape=(n_folds, 4)
            means = metrics_arr.mean(axis=0)
            stds = metrics_arr.std(axis=0, ddof=1)

            print(
                f"  >> {ds_name} / {model_name}: "
                f"Acc={means[0]:.3f}±{stds[0]:.3f}, "
                f"Sens={means[1]:.3f}±{stds[1]:.3f}, "
                f"Spec={means[2]:.3f}±{stds[2]:.3f}, "
                f"AUC={means[3]:.3f}±{stds[3]:.3f}"
            )

            summary_stats.append(
                {
                    "Dataset": ds_name,
                    "Model": model_name,
                    "Mean_Accuracy": means[0],
                    "Std_Accuracy": stds[0],
                    "Mean_Sensitivity": means[1],
                    "Std_Sensitivity": stds[1],
                    "Mean_Specificity": means[2],
                    "Std_Specificity": stds[2],
                    "Mean_AUC": means[3],
                    "Std_AUC": stds[3],
                }
            )

    # --------------------------------------------------------------
    # 4. Save fold-wise metrics and summary
    # --------------------------------------------------------------
    if rows_folds:
        df_folds = pd.DataFrame(rows_folds)
        folds_csv = REPORT_DIR / "baseline_folds.csv"
        df_folds.to_csv(folds_csv, index=False)
        print(f"\nSaved per-fold nested CV metrics to {folds_csv}")
    else:
        print("[WARN] No per-fold metrics collected.")

    if summary_stats:
        df_summary = pd.DataFrame(summary_stats)
        summary_csv = REPORT_DIR / "baseline_result_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"Saved summary metrics to {summary_csv}\n")
        print("=== Baseline summary ===")
        print(df_summary)
    else:
        print("[WARN] No summary statistics computed.")

    # --------------------------------------------------------------
    # 5. Confusion matrices from aggregated outer-fold predictions
    #    Every subject was predicted exactly once on a held-out fold.
    # --------------------------------------------------------------
    print("\nPlotting confusion matrices from aggregated outer-fold predictions ...")

    for ds_name, (X, y_vec) in datasets.items():
        if ds_name not in oof_preds:
            continue
        for model_name, (y_true_oof, y_pred_oof) in oof_preds[ds_name].items():
            out_path = REPORT_FIGS / f"cm_opt_{ds_name}_{model_name}.png"
            title = f"{ds_name} – {model_name}"
            plot_confusion_matrix(y_true_oof, y_pred_oof, title, out_path)

    print("\nDONE. Baseline artifacts written to 'reports/' and 'reports/figs/'.")


if __name__ == "__main__":
    main()
