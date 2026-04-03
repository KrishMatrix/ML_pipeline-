# 🏠 California Housing Price Prediction

> End-to-end ML pipeline for predicting median house values — from EDA and model comparison to a persistent train-once-predict-forever inference system using **joblib**.  
> Built as a learning project following the *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* workflow.

---

## 📌 What This Project Does

This project takes the [California Housing dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) and builds a complete ML workflow in two phases:

1. **`model_gurgau.ipynb`** — Explore the data, build preprocessing pipelines, and compare 3 regression models to find the best one.
2. **`joblib.ipynb`** — Take the winning model (RandomForest), wrap it into a production-style pipeline that trains once, serializes to disk, and loads instantly for future predictions.

---

## 🧠 Key Concepts Practiced

| Concept | Where It's Used |
|---|---|
| **Stratified Sampling** | `StratifiedShuffleSplit` on `median_income` bins to avoid sampling bias |
| **sklearn Pipelines** | `Pipeline` + `ColumnTransformer` to chain imputation → scaling → encoding |
| **Model Comparison** | LinearRegression vs DecisionTree vs RandomForest (RMSE + 10-fold CV) |
| **Overfitting Detection** | DecisionTree got 0.0 train RMSE but ~69k CV RMSE → classic overfit |
| **Cross-Validation** | 10-fold CV with `neg_mean_squared_error` to get honest error estimates |
| **Model Persistence** | `joblib.dump()` / `joblib.load()` to serialize model & pipeline to `.pkl` |
| **Train/Inference Toggle** | `os.path.exists()` check — trains if no `.pkl` found, predicts if found |

---

## 📊 Dataset Overview

The California Housing dataset contains **20,640 districts** with these features:

| Feature | Type | Description |
|---|---|---|
| `longitude` | float | District longitude |
| `latitude` | float | District latitude |
| `housing_median_age` | float | Median age of houses in the district |
| `total_rooms` | float | Total rooms in the district |
| `total_bedrooms` | float | Total bedrooms (has missing values!) |
| `population` | float | District population |
| `households` | float | Number of households |
| `median_income` | float | Median income (scaled, ~0.5 to 15) |
| `ocean_proximity` | string | Categorical — NEAR BAY, INLAND, NEAR OCEAN, <1H OCEAN, ISLAND |
| **`median_house_value`** | float | **Target variable** — what we predict |

---

## 🗂️ Project Structure

```
california-housing-prediction/
│
├── model_gurgau.ipynb        # EDA + model comparison (LinReg, DecTree, RandomForest)
├── joblib.ipynb              # Production pipeline — train OR predict in one run
├── housing.csv               # Raw dataset (California Housing)
│
├── model_gurgau.pkl          # [Generated] Trained RandomForest model
├── pipeline_gurgau.pkl       # [Generated] Fitted preprocessing pipeline
├── input_data.csv            # [Generated] Held-out test set (20%)
├── predictions.csv           # [Generated] Test set + predicted values
├── final_predictions.csv     # [Generated] Side-by-side actual vs predicted
│
├── .gitignore
└── README.md
```

> Files marked `[Generated]` are created automatically when you run the notebooks.

---

## 📓 Notebook 1: `model_gurgau.ipynb` — EDA & Model Comparison

This is the **experimentation notebook** where models were tested before picking the best one.

### Step-by-step walkthrough

| Cell | What It Does | Key Detail |
|---|---|---|
| **1** | Imports | numpy, pandas, sklearn (Pipeline, Imputer, Scaler, Encoder, models) |
| **2** | Load data | `pd.read_csv("housing.csv")` → 20,640 rows × 10 columns |
| **3** | Create income strata | `pd.cut(median_income)` into 5 bins → `income_cat` column for stratified splitting |
| **4** | Stratified train/test split | 80/20 split using `StratifiedShuffleSplit` on `income_cat` → 16,512 train / 4,128 test |
| **5** | Separate features & labels | `housing_labels` = `median_house_value`, drop target from features |
| **6** | Identify column types | Numeric: `longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, income_cat` · Categorical: `ocean_proximity` |
| **7** | Build sub-pipelines | **Numeric:** `SimpleImputer(median)` → `StandardScaler` · **Categorical:** `SimpleImputer(most_frequent)` → `OneHotEncoder` |
| **8** | Full ColumnTransformer | Combines num + cat pipelines → `fit_transform(housing)` → sparse matrix output |
| **9** | Linear Regression | Train RMSE: **68,867** · 10-fold CV MSE scores computed |
| **10** | Decision Tree | Train RMSE: **0.0** (overfitting!) · CV scores much worse → model memorized training data |
| **11** | Random Forest | Train RMSE: **18,443** · CV scores best of all three → **selected as final model** |
| **12** | Note to self | "joblib is used for saving the model, because I can't train the model again and again" |

### Model Comparison Results

```
┌──────────────────────┬────────────────┬──────────────────────────────┐
│ Model                │ Train RMSE ($) │ Verdict                      │
├──────────────────────┼────────────────┼──────────────────────────────┤
│ Linear Regression    │ 68,867         │ Underfitting — too simple     │
│ Decision Tree        │ 0              │ Overfitting — memorized data  │
│ Random Forest        │ 18,443         │ ✅ Best — generalized well    │
└──────────────────────┴────────────────┴──────────────────────────────┘
```

**Why RandomForest won:** It achieved the lowest train RMSE *and* the best cross-validation scores, meaning it generalized well to unseen data — unlike the Decision Tree which got 0 training error but fell apart on CV folds.

---

## 📓 Notebook 2: `joblib.ipynb` — Production Pipeline

The main workhorse. Takes the winning RandomForest model and wraps it into a **train-or-load** system.

### Step-by-step walkthrough

| Cell | What It Does |
|---|---|
| **1** | Imports (adds `os`, `joblib`, `RandomForestRegressor` to the stack) |
| **2** | **Core logic** — checks if `.pkl` files exist. If NO → full training pipeline + save. If YES → load + predict. |
| **3** | Merge predictions with original test data → `final_predictions.csv` |

### The train-or-load pattern

```python
if not os.path.exists(MODEL_FILE) or not os.path.exists(PIPELINE_FILE):
    # TRAIN: load data → split → preprocess → fit model → save .pkl
else:
    # PREDICT: load .pkl → transform new data → predict → save CSV
```

---

## 🔄 Pipeline Flow Diagrams

### Phase 1 — Training (first run, no `.pkl` files exist)

```
housing.csv
    │
    ▼
Stratified Split (80/20 by median_income bins)
    │
    ├──► Train set (16,512 rows)
    │       │
    │       ▼
    │   Preprocessing Pipeline (ColumnTransformer)
    │       ├── Numeric cols  → SimpleImputer(median) → StandardScaler
    │       └── Categorical   → SimpleImputer(mode)   → OneHotEncoder
    │       │
    │       ▼
    │   RandomForestRegressor(random_state=42).fit()
    │       │
    │       ├──► model_gurgau.pkl     (joblib.dump)
    │       └──► pipeline_gurgau.pkl  (joblib.dump)
    │
    └──► Test set (4,128 rows) → input_data.csv
```

### Phase 2 — Inference (subsequent runs, `.pkl` files found)

```
model_gurgau.pkl + pipeline_gurgau.pkl  (joblib.load)
    │
    ▼
input_data.csv → pipeline.transform() → model.predict()
    │
    ├──► predictions.csv            (test data + predicted column)
    └──► final_predictions.csv      (original test data + predicted_median_house_value)
```

---

## 🧪 Preprocessing Pipeline Details

```
ColumnTransformer
├── "num" (numeric columns)
│   └── Pipeline
│       ├── SimpleImputer(strategy="median")    ← fills NaN with column median
│       └── StandardScaler()                     ← zero-mean, unit-variance
│
└── "cat" (ocean_proximity)
    └── Pipeline
        ├── SimpleImputer(strategy="most_frequent")  ← fills NaN with mode
        └── OneHotEncoder()                           ← binary dummy columns
```

**Why save the pipeline too?** New/test data needs the *exact same* transformations — same medians, same scaling factors, same encoding categories — that were fitted on the training data. Without the saved pipeline, you'd need the original training set every time.

---

## 🚀 Getting Started

### Prerequisites

```bash
python -m pip install numpy pandas scikit-learn joblib jupyter
```

### Run the experimentation notebook

```bash
jupyter notebook model_gurgau.ipynb
# Run all cells → see model comparison results
```

### Run the production pipeline

```bash
jupyter notebook joblib.ipynb
# First run  → trains model, creates .pkl files
# Second run → loads model, generates predictions
```

### Retrain from scratch

```bash
rm model_gurgau.pkl pipeline_gurgau.pkl
# Next run of joblib.ipynb will retrain
```

### Expected Console Output

| Run | Output |
|---|---|
| First run (training) | `model is trained and saved successfully` |
| Subsequent runs (inference) | `model and pipeline are loaded successfully` |

---

## 🐛 Bugs Fixed During Development

### 1. `DataFrame.append()` removed in pandas 2.0+

**Error:** `AttributeError: 'DataFrame' object has no attribute 'append'`

```python
# ❌ Old (broken in pandas 2.0+)
input_data = input_data.append(predictions_data["median_house_value"], ignore_index=True)

# ✅ Fix — direct column assignment
input_data["predicted_median_house_value"] = predictions_data["median_house_value"]
```

### 2. `income_cat` leaking into numeric features (model_gurgau.ipynb)

In the experimentation notebook, `income_cat` (the stratification column) was not dropped before training, so it ended up as a numeric feature. The production notebook (`joblib.ipynb`) fixes this by explicitly dropping it:

```python
housing = housing.drop("income_cat", axis=1)
```

---

## 📝 Notes & Learnings

- **Stratified split matters:** Random splits can over/under-represent income groups. `StratifiedShuffleSplit` ensures the test set mirrors the real income distribution.
- **Pipeline = reproducibility:** Wrapping all preprocessing in a `ColumnTransformer` means you can't accidentally apply different transformations to train vs test data.
- **Decision Tree overfitting is dramatic:** 0.0 train RMSE looks amazing until you see the cross-validation scores explode — a textbook example of why CV matters.
- **joblib > pickle for sklearn:** `joblib` is more efficient for objects containing large NumPy arrays (like fitted model internals).
- **The `.pkl` check pattern** (`if not os.path.exists(...)`) is a simple way to avoid retraining — useful in notebooks where you re-run cells frequently.
- **Always save the pipeline with the model.** A model alone is useless if you can't preprocess new data the same way.

---

## 🔮 Possible Next Steps

- [ ] Evaluate on the actual test set (compare predicted vs true `median_house_value`)
- [ ] Hyperparameter tuning with `GridSearchCV` or `RandomizedSearchCV`
- [ ] Add feature importance visualization from RandomForest
- [ ] Try `GradientBoostingRegressor` or `XGBRegressor`
- [ ] Build a Flask/Streamlit app for interactive predictions
- [ ] Add proper logging instead of `print()` statements

---

## 📦 Suggested `.gitignore`

```gitignore
# Generated model files (large, regenerable)
*.pkl

# Generated data files
input_data.csv
predictions.csv
final_predictions.csv

# Jupyter checkpoints
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc

# OS
.DS_Store
```

> **Tip:** If you want collaborators to run inference without retraining, commit the `.pkl` files. If the repo is code-only, add them to `.gitignore` since they're regenerable.

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.13.7 | Runtime |
| pandas | 2.0+ | Data loading & manipulation |
| NumPy | latest | Numerical operations |
| scikit-learn | latest | Pipelines, models, metrics, CV |
| joblib | latest | Model & pipeline serialization |
| Jupyter | latest | Interactive notebook development |
