# California Housing Intelligence

An end-to-end machine learning portfolio project that studies one public housing dataset through three complementary tasks: **predicting home values (regression), assigning market tiers (classification), and discovering regional housing segments (clustering).** It also includes exploratory analysis, preprocessing pipelines, cross-validation, model comparison, feature importance, reproducible artifacts, and practical caveats.

## Why this project

The California Housing dataset is public and small enough to run on a laptop, yet rich enough to demonstrate a complete applied ML workflow. The source contains 20,640 California census block groups and eight demographic/geographic features. The target, `MedHouseVal`, is median house value in units of $100,000. It is an old benchmark dataset, so this project is an educational portfolio exercise—not a current valuation product.

## Tasks

| Task | Question | Approach | Main metrics |
|---|---|---|---|
| Regression | How well can the available block-group features estimate median house value? | Median baseline, Random Forest, and Histogram Gradient Boosting; 5-fold shuffled CV selects the model | MAE, RMSE, R² |
| Classification | Which of three relative market tiers does a block group belong to? | Random Forest classifier; tier cutoffs are computed from training targets only | Macro/weighted F1, accuracy, confusion matrix |
| Clustering | Which housing areas have similar demographic and geographic profiles? | StandardScaler + KMeans; silhouette scores compare k=2…8 | Silhouette score, cluster sizes and profiles |

Tier boundaries are the training-set tertiles, so “Value”, “Mid-market”, and “Premium” are relative labels for this dataset rather than universal affordability thresholds. The target is never included as a clustering input. Scaling and imputation are encapsulated in model pipelines. The random holdout is suitable for a benchmark; spatially nearby observations can be similar, so it may overstate generalization to entirely new regions.

## Quick start

Requires Python 3.10 or newer and internet access the first time the dataset is fetched.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python -m pip install -e .
housing-ml
```

Or run without installing the command-line entry point:

```bash
PYTHONPATH=src python -m housing_ml.cli --output artifacts --seed 42
```

The first run downloads the dataset through scikit-learn's public dataset loader and caches it in the normal scikit-learn data cache. Generated results stay in `artifacts/` and are excluded from Git.

Options:

```text
housing-ml --help
housing-ml --output artifacts --seed 17
housing-ml --skip-clustering
```

## Generated artifacts

- `metrics.json` — dataset details, holdout metrics, CV scores, tier boundaries, and clustering selection.
- `eda_overview.png` — target distribution and sampled geographic value map.
- `clusters_map.png`, `cluster_profiles.csv` — unsupervised segment map and interpretable feature averages.
- `regression_feature_importance.csv` — held-out permutation importance for the CV-selected regressor.
- `classification_confusion_matrix.csv` — actual tier by predicted tier.
- `*.joblib` — fitted regression/classification models and clustering model with scaler. Load only artifacts you trust; joblib files can execute code when loaded.

## Concepts demonstrated

- Problem framing across supervised and unsupervised learning
- Train/test split, deterministic seeds, 5-fold cross-validation, and simple baseline comparison
- Missing-value imputation and feature scaling with scikit-learn pipelines
- Regression metrics with distinct interpretations (absolute error, squared error, explained variance)
- Multiclass classification, class weighting, macro vs. weighted F1, and confusion matrix analysis
- Target-derived labels with training-only thresholds to avoid test-set leakage
- KMeans, standardization, silhouette-based k selection, cluster profiling, and visualization
- Held-out permutation importance, serialized models, a CLI, and machine-readable experiment results
- Data limitations, geographic dependence, capped target values, and responsible interpretation

## Suggested analysis questions

1. Does the boosted model improve on the median and Random Forest baselines on the held-out set?
2. Which features matter most under permutation importance, and does that match domain intuition?
3. Which classes are easiest to confuse? Would a binary premium/non-premium task be more useful?
4. How do cluster profiles differ in median income, population density, and location?
5. How do metrics change under a geographic holdout (for example, holding out a region rather than randomly sampling rows)?

## Resume-ready summary

Use only after running the project, and replace bracketed values with the results in `artifacts/metrics.json`:

> Built a reproducible California housing ML workflow comparing Random Forest and gradient-boosted regression with 5-fold cross-validation; achieved **[RMSE]** test RMSE. Added a train-only price-tier classifier (**[macro-F1]** macro-F1) and KMeans segmentation with silhouette-guided cluster selection, interpretable profiles, and visual reporting.

Interview talking points: explain why tier boundaries are learned from the training targets, why clustering is kept separate from supervised labels, why the target's top end is capped, and how you would evaluate transfer to a geographically separate region.

## Dataset and responsible use

The data was collected from 1990 U.S. Census block groups. Features include location, median income, housing age, rooms, bedrooms, population, and household count. Its age, ecological (area-level) aggregation, and geographic coverage limit what it can say about today's individual homes or people. The historical target is top-coded near $500,000. Do not use these predictions for lending, appraisal, tenant screening, or other decisions about individuals.

Dataset loader documentation: [scikit-learn California Housing](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html). Original data source: [StatLib California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).

## Project layout

```text
src/housing_ml/
  data.py          dataset loading and basic validation
  features.py      train-only target tiering
  pipeline.py      model definitions
  experiment.py    full experiment and artifact generation
  cli.py           command-line interface
```
