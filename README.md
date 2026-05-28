# Auto MPG — Regression & Clustering 

## Overview

An end-to-end machine learning project combining regression and unsupervised clustering on vehicle fuel-efficiency data. The workflow covers dataset inspection, train/test splitting with leak-free scaling, comparison of three regression models, 5-fold cross-validation, KMeans cluster selection, and DBSCAN comparison.

**Data source:** [Auto MPG Dataset — Kaggle (YasserH)](https://www.kaggle.com/datasets/yasserh/auto-mpg-dataset)

---

## Dataset

| Property | Detail |
|---|---|
| File | `auto_mpg_clean.csv` |
| Missing values | None (rows with missing horsepower removed in preprocessing) |
| Duplicate rows | None |
| Target column | `mpg` (miles per gallon: continuous) |

### Features

| Column | Description |
|---|---|
| `cylinders` | Number of engine cylinders |
| `displacement` | Engine displacement (cubic inches) |
| `horsepower` | Engine horsepower |
| `weight` | Vehicle weight (lbs) |
| `acceleration` | 0–60 mph acceleration time (seconds) |
| `model_year` | Model year (last two digits) |

### Preprocessing (done prior to this project)

- Rows with missing `horsepower` values were removed.
- The `model-year` column was renamed to `model_year`.
- No additional cleaning was required.

---

## Workflow

### Part 1: Regression

**Setup**
- Target: `y = df['mpg']`; Features: `X = df.drop(columns=['mpg'])`
- 80/20 stratified train/test split (`random_state=42`)
- `StandardScaler` fitted on `X_train` only, then applied to both splits

**Models trained**

| Model | Data used | Settings |
|---|---|---|
| `LinearRegression` | Scaled | Default |
| `KNeighborsRegressor` | Scaled | `n_neighbors=7` |
| `DecisionTreeRegressor` | Unscaled | `max_depth=6, random_state=42` |

**Evaluation:** MAE, RMSE, and R² reported on the test set for all three models.

**Visualisation:** Actual vs. predicted `mpg` scatter plot for the Decision Tree model.

**Cross-validation:** 5-fold CV applied to the Decision Tree using `KFold(n_splits=5, shuffle=True, random_state=42)` with `scoring='r2'`; individual fold R² scores and mean CV R² printed.

---

### Part 2: Clustering

**Features used:** `horsepower` and `weight` (scaled with `StandardScaler` before clustering).

**KMeans selection:** Evaluated for k = 2, 3, 4, 5 using inertia and silhouette score; best k selected from the table.

**Final KMeans model:** `KMeans(n_clusters=2, random_state=42, n_init=20)`; cluster sizes and mean profile table (horsepower, weight, mpg per cluster) produced.

**DBSCAN:** k-distance plot (6th nearest neighbour) used to identify the elbow; fitted with `DBSCAN(eps=0.28, min_samples=6)`.

---

## Key Findings

**Regression:** Checking MAE, RMSE, and R² together matters because each metric captures a different aspect of model error. MAE reflects average absolute error, RMSE penalises large errors more heavily, and R² shows how much variance the model explains. Relying on any single metric can be misleading.

**Clustering:** KMeans assigns every point to a cluster, producing more compact groups and a higher silhouette score (0.616). DBSCAN identifies 27 noise points and excludes them, which lowers the silhouette score (0.536) but reveals additional structure by distinguishing outliers that KMeans would otherwise force into a cluster.


### Clustering comparison

| Method | Clusters Found | Noise Points | Silhouette Score |
|---|---|---|---|
| KMeans | 2 | 0 | 0.6162 |
| DBSCAN | 3 | 27 | 0.5364 |
