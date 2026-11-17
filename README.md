# Megaline Mobile Plan Classification

Supervised machine learning classification project to recommend optimal mobile plans (**Smart** vs. **Ultra**) for Megaline subscribers based on monthly usage behavior.

---

## Business Context & Objective

Megaline, a mobile telecommunications provider, aims to improve its customer plan recommendations by identifying subscribers whose usage patterns align with its two current offerings: **Smart** and **Ultra**. Using historical monthly subscriber behavior data (`calls`, `minutes`, `messages`, and `mb_used`), the objective is to build a binary classification model that accurately predicts the right plan for each customer.

- **Primary Metric:** Accuracy
- **Acceptance Threshold:** >= 0.75 (`75.00%`) on an independent test set

---

## Dataset Overview

The dataset (`Datasets/users_behavior.csv`) contains **3,214** monthly subscriber records with **0 missing values** and **0 duplicates**:

| Feature | Type | Description |
|---|---|---|
| `calls` | `float64` | Number of calls made during the month |
| `minutes` | `float64` | Total call duration in minutes |
| `messages` | `float64` | Number of text messages sent |
| `mb_used` | `float64` | Mobile internet traffic consumed in MB |
| `is_ultra` | `int64` | Target variable: `0` = Smart (`69.35%`), `1` = Ultra (`30.65%`) |

---

## Methodology & Workflow

1. **Exploratory Data Analysis:** Verified data integrity, summary statistics, and target class balance (`2,229` Smart vs. `985` Ultra).
2. **Stratified Data Splitting:** Partitioned data into **60% Training** (`1,928` rows), **20% Validation** (`643` rows), and **20% Test** (`643` rows) using `RANDOM_STATE = 12345`.
3. **Model Training & Hyperparameter Tuning:** Evaluated `DecisionTreeClassifier` (`max_depth` 1–15), `RandomForestClassifier` (`n_estimators` 10–100, `max_depth` 2–None), and standardized `LogisticRegression` (`StandardScaler` + `Pipeline`).
4. **Independent Test Evaluation:** Assessed the winning model on the held-out test set via accuracy, confusion matrix, and precision/recall/F1 metrics.
5. **Sanity Check & Diagnostics:** Benchmarked against a `DummyClassifier` (`most_frequent`) baseline and analyzed prediction probabilities, subscriber usage profiles, and permutation feature importance.

---

## Model Comparison & Key Results

| Model | Best Hyperparameters | Validation Accuracy | Test Accuracy |
|---|---|---|---|
| **Random Forest (Selected)** | `n_estimators=10`, `max_depth=8` | **0.8320** | **0.8103** |
| Decision Tree | `max_depth=5` | 0.8165 | — |
| Logistic Regression | `StandardScaler` + `max_iter=1000` | 0.7558 | — |
| Dummy Baseline | `strategy='most_frequent'` | — | 0.6936 |

### Key Takeaways
- **Requirement Met:** The selected Random Forest model achieved **81.03% test accuracy**, surpassing the `75.00%` project threshold by **+6.03 percentage points** and outperforming the majority-class baseline by **+11.66 percentage points**.
- **Strong Generalization:** Validation-to-test accuracy gap is only **2.18 percentage points** (`0.8320` vs. `0.8103`).
- **Primary Usage Driver:** Permutation importance shows mobile internet usage (`mb_used`) is the most influential feature (`-11.59%` accuracy when shuffled), followed by `minutes` (`-6.77%`), `messages` (`-5.40%`), and `calls` (`-4.14%`).
- **Class Performance:** High precision across both classes (`81.03%` Smart, `80.99%` Ultra), with `94.84%` recall on Smart subscribers and `49.75%` recall on Ultra subscribers due to usage overlap among moderate-consumption Ultra users.

---

## Repository Structure

```text
megaline-mobile-plan-classification/
├── Datasets/
│   └── users_behavior.csv              # Monthly subscriber usage dataset
├── megaline_plan_classification.ipynb  # Complete analysis, modeling & diagnostics notebook
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Standard Python/Jupyter ignore rules
└── README.md                           # Project documentation