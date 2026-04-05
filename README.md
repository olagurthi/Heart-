# 🫀 Heart Disease Prediction

> A supervised binary classification notebook comparing four ML models on clinical heart data — with hyperparameter tuning, cross-validation, and full evaluation metrics.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-1.3+-150458?style=flat&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)

---

## 📋 Table of Contents

- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Notebook Walkthrough](#-notebook-walkthrough)
- [Results](#-results)
- [Key Observations](#-key-observations)
- [Evaluation Metrics](#-evaluation-metrics)

---

## 📊 Dataset

**File:** `heart.csv` — 918 patients, 11 features, 1 binary target

| Property | Value |
|---|---|
| Rows | 918 |
| Features | 11 |
| Target | `HeartDisease` (0 = no disease · 1 = disease) |
| Class balance | 55% positive · 45% negative |
| Missing values | None |

### Features

| Feature | Type | Description |
|---|---|---|
| `Age` | Numeric | Patient age in years |
| `Sex` | Categorical | M = male, F = female |
| `ChestPainType` | Categorical | ATA · ASY · NAP · TA |
| `RestingBP` | Numeric | Resting blood pressure (mm Hg) |
| `Cholesterol` | Numeric | Serum cholesterol (mm/dl) |
| `FastingBS` | Categorical | Blood sugar > 120 mg/dl (1 = true) |
| `RestingECG` | Categorical | Normal · ST · LVH |
| `MaxHR` | Numeric | Maximum heart rate achieved |
| `ExerciseAngina` | Categorical | Exercise-induced angina (Y/N) |
| `Oldpeak` | Numeric | ST depression induced by exercise |
| `ST_Slope` | Categorical | Slope of peak exercise ST segment (Up · Flat · Down) |

---

## 📁 Project Structure

```
heart-disease-ml/
├── heart.csv               ← dataset (918 rows)
├── heart_disease_ml.py     ← main notebook
└── README.md               ← this file
```

---

## ⚙️ Installation

```bash
# Install all dependencies
pip install pandas numpy matplotlib scikit-learn

# Launch Jupyter
jupyter notebook
```

Then open `heart_disease_ml.py` and run **Kernel → Restart & Run All**.

> The `# %%` cell markers are recognised natively by Jupyter, JupyterLab, and VS Code's Jupyter extension.

---

## 📓 Notebook Walkthrough

### 1 · Load & Explore
Reads `heart.csv`, prints shape and data types, checks for missing values, and plots target class balance alongside an age histogram split by class.

---

### 2 · Preprocessing

| Step | Method | Applied to |
|---|---|---|
| Binary encoding | `LabelEncoder` | `Sex`, `ExerciseAngina` |
| Multi-class encoding | `pd.get_dummies()` | `ChestPainType`, `RestingECG`, `ST_Slope` |
| Train/test split | `train_test_split(test_size=0.2, stratify=y)` | All models |
| Feature scaling | `StandardScaler` | Linear & Logistic Regression only |

> **Note:** Decision Tree and Naive Bayes do not require feature scaling — they operate on value splits and probability distributions, not distances.

---

### 3 · Decision Tree

**3a — Hyperparameter tuning (`max_depth`)**

Tests depths 1–20 and plots train vs. test accuracy to find the overfitting crossover point. Best depth is selected automatically.

```
Too shallow  →  underfitting (misses patterns)
Too deep     →  overfitting  (memorises training data)
```

**3b — Cross-validation**

Runs `StratifiedKFold(n_splits=5)` with the best depth. Stratified folds preserve the 55/45 class ratio in every split. Reports per-fold scores, mean, and standard deviation.

**3c — Final model outputs**
- Classification report
- Decision tree plot (top 3 levels via `plot_tree()`)
- Feature importance bar chart
- Confusion matrix

---

### 4 · Linear Regression *(baseline)*

Fits a standard linear regressor and thresholds predictions at 0.5 to produce binary labels. Used only as a performance floor — predictions can fall outside [0, 1], which is its core limitation. Includes a coefficient plot.

---

### 5 · Logistic Regression

**5a — GridSearchCV tuning (`C`)**

Searches 15 values of regularisation strength `C` on a log scale from 10⁻⁵ to 10⁸ using 5-fold cross-validation. Plots mean CV accuracy vs. C with a shaded confidence band.

```
Small C  →  strong regularisation  →  simpler model
Large C  →  weak regularisation   →  complex model (overfitting risk)
```

> 🏆 **Best result:** `C = 0.006105` · CV accuracy = `0.853`

**5b — Final model outputs**
- Classification report
- Odds ratio bar chart (exponentiated coefficients)
- ROC curve

---

### 6 · Naive Bayes

Trains `GaussianNB`, assuming numeric features follow a Gaussian distribution per class. No scaling required. Produces a classification report and confusion matrix.

---

### 7 · Model Comparison

Collects accuracy and AUC-ROC for all four models and produces:
- Ranked summary table
- Side-by-side accuracy and AUC-ROC bar charts
- Overlaid ROC curves for all models on one plot

---

## 📈 Results

| Model | Accuracy | AUC-ROC | Scaling | Best for |
|---|---|---|---|---|
| ✅ Logistic Regression | ~87% | ~0.93 | Required | Accuracy + calibrated probabilities |
| 🌳 Decision Tree | ~83% | ~0.88 | Not needed | Interpretability + clinical rules |
| 📊 Naive Bayes | ~82% | ~0.88 | Not needed | Speed + small datasets |
| 📉 Linear Regression | ~80% | ~0.87 | Required | Baseline comparison only |

---

## 🔍 Key Observations

**1 — Top features align with clinical knowledge**
`ST_Slope`, `ChestPainType`, and `Oldpeak` are consistently the most important features — well-known ECG and symptom indicators of cardiac stress during exercise.

**2 — Strong regularisation was needed for Logistic Regression**
GridSearchCV tuned `C` to `0.006105` (best CV score: `0.853`) — far below the default of 1.0. This indicates strong regularisation was needed, likely due to multicollinearity introduced by the one-hot encoded categorical columns.

**3 — Decision Tree wins on interpretability**
Despite slightly lower accuracy, its rules can be read directly by a clinician — e.g. *"flat ST slope AND Oldpeak > 1.5 → predict disease"*. In a medical setting this often matters more than a 2–3% accuracy difference.

**4 — Naive Bayes punches above its weight**
Performed competitively despite the naive independence assumption, confirming that individual features carry enough signal about heart disease risk on their own.

**5 — Recall matters more than accuracy here**
For clinical deployment, a missed diagnosis (false negative) is far more costly than a false alarm. The decision threshold should be tuned to maximise Recall for the Disease class, not raw accuracy.

---

## 📐 Evaluation Metrics

| Metric | Why it matters |
|---|---|
| **Accuracy** | Overall % correct — reliable here since classes are near-balanced (55/45) |
| **Precision** | Of all predicted disease cases, how many were actually disease? |
| **Recall** | Of all actual disease cases, how many did the model catch? *(critical in medical diagnosis)* |
| **F1 Score** | Harmonic mean of precision and recall — good single summary metric |
| **AUC-ROC** | Threshold-independent ranking quality across all possible decision boundaries |

---

<div align="center">
  <sub>Built with scikit-learn · pandas · matplotlib</sub>
</div>
