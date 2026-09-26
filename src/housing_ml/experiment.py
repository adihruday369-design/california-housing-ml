"""Reproducible training, comparison, clustering, and reporting workflow."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import load_housing
from .features import TIERS, assign_tiers, tier_edges
from .pipeline import classifier, regression_models


def _json_safe(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def _save_eda(X: pd.DataFrame, y: pd.Series, out: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.histplot(y, bins=45, ax=axes[0], color="#2563eb")
    axes[0].set(title="Median house value distribution", xlabel="Value (units of $100,000)")
    sample = pd.concat([X, y], axis=1).sample(min(4500, len(X)), random_state=42)
    sns.scatterplot(data=sample, x="Longitude", y="Latitude", hue="MedHouseVal",
        palette="viridis", s=10, alpha=0.55, legend=False, ax=axes[1])
    axes[1].set(title="Geographic pattern in the sample", xlabel="Longitude", ylabel="Latitude")
    fig.tight_layout()
    fig.savefig(out / "eda_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _cluster_analysis(X: pd.DataFrame, out: Path, seed: int) -> dict:
    # Scale because raw feature magnitudes differ; no target is used for clustering.
    scaler = StandardScaler()
    scaled = scaler.fit_transform(X)
    k_values = range(2, 9)
    scores = {}
    for k in k_values:
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(scaled)
        scores[str(k)] = float(silhouette_score(scaled, labels, sample_size=4000, random_state=seed))
    chosen_k = int(max(scores, key=scores.get))
    model = KMeans(n_clusters=chosen_k, n_init=20, random_state=seed)
    labels = model.fit_predict(scaled)
    labeled = X.copy()
    labeled["Cluster"] = labels
    profile = labeled.groupby("Cluster").mean(numeric_only=True)
    sizes = labeled["Cluster"].value_counts().sort_index()
    profile.insert(0, "row_count", sizes)
    profile.to_csv(out / "cluster_profiles.csv")
    joblib.dump({"scaler": scaler, "model": model, "features": list(X.columns)}, out / "cluster_model.joblib")
    # Compact 2-D geographic view to make the segments interpretable.
    fig, ax = plt.subplots(figsize=(10, 6))
    plot = X[["Longitude", "Latitude"]].copy()
    plot["Cluster"] = labels.astype(str)
    sns.scatterplot(data=plot.sample(min(6000, len(plot)), random_state=seed),
        x="Longitude", y="Latitude", hue="Cluster", palette="tab10", s=9, alpha=0.65, ax=ax)
    ax.set_title(f"Unsupervised housing segments (KMeans, k={chosen_k})")
    fig.tight_layout()
    fig.savefig(out / "clusters_map.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"selected_k": chosen_k, "silhouette_by_k": scores, "cluster_sizes": sizes.to_dict()}


def run_experiment(output_dir: Path, seed: int = 42, clustering: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    X, y = load_housing()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    _save_eda(X, y, output_dir)

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    regression_results = {}
    trained_regressors = {}
    for name, model in regression_models(seed).items():
        if name == "baseline_median":
            pred = np.full(len(y_test), y_train.median())
            cv_rmse = None
        else:
            scores = cross_validate(model, X_train, y_train, cv=cv,
                scoring={"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"},
                n_jobs=1)
            cv_rmse = float(-scores["test_rmse"].mean())
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            trained_regressors[name] = model
            joblib.dump(model, output_dir / f"{name}.joblib")
        regression_results[name] = {
            "test_mae": float(mean_absolute_error(y_test, pred)),
            "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "test_r2": float(r2_score(y_test, pred)), "cv_rmse_mean": cv_rmse,
        }

    best_name = min((n for n in regression_results if n != "baseline_median"),
        key=lambda n: regression_results[n]["cv_rmse_mean"])
    best_regressor = trained_regressors[best_name]
    perm = permutation_importance(best_regressor, X_test, y_test, n_repeats=5,
        random_state=seed, scoring="neg_root_mean_squared_error", n_jobs=-1)
    importance = pd.DataFrame({"feature": X.columns, "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(output_dir / "regression_feature_importance.csv", index=False)

    edges = tier_edges(y_train)
    y_train_tier, y_test_tier = assign_tiers(y_train, edges), assign_tiers(y_test, edges)
    clf = classifier(seed)
    clf.fit(X_train, y_train_tier)
    tier_pred = clf.predict(X_test)
    class_results = {
        "accuracy": float(accuracy_score(y_test_tier, tier_pred)),
        "macro_f1": float(classification_report(y_test_tier, tier_pred, output_dict=True, zero_division=0)["macro avg"]["f1-score"]),
        "weighted_f1": float(classification_report(y_test_tier, tier_pred, output_dict=True, zero_division=0)["weighted avg"]["f1-score"]),
        "confusion_matrix": confusion_matrix(y_test_tier, tier_pred, labels=TIERS).tolist(),
        "classification_report": classification_report(y_test_tier, tier_pred, labels=TIERS,
            target_names=TIERS, output_dict=True, zero_division=0),
        "tier_edges_in_100k_dollars": edges.tolist(),
    }
    joblib.dump({"model": clf, "tier_edges": edges, "tiers": TIERS, "features": list(X.columns)},
        output_dir / "price_tier_classifier.joblib")
    pd.DataFrame(confusion_matrix(y_test_tier, tier_pred, labels=TIERS),
        index=TIERS, columns=TIERS).to_csv(output_dir / "classification_confusion_matrix.csv")

    results = {
        "dataset": {"name": "California Housing", "source": "scikit-learn fetch_california_housing / StatLib",
            "rows": int(len(X)), "features": int(X.shape[1]), "target_unit": "$100,000", "random_seed": seed},
        "regression": {"models": regression_results, "selected_model": best_name},
        "classification": class_results,
        "clustering": _cluster_analysis(X, output_dir, seed) if clustering else {"skipped": True},
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, default=_json_safe) + "\n")
    return results
