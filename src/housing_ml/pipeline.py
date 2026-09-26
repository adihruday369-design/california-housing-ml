"""Model construction and evaluation helpers."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def regression_models(seed: int) -> dict:
    return {
        "baseline_median": Pipeline([("impute", SimpleImputer(strategy="median"))]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                max_features=0.85, n_jobs=-1, random_state=seed)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                l2_regularization=1.0, random_state=seed)),
        ]),
    }


def classifier(seed: int) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
    ])
