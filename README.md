# Heart Disease Prediction — Machine Learning Notebook

A supervised binary classification project that predicts the presence of heart disease using four machine learning models, with hyperparameter tuning and cross-validation applied throughout.

---

## Dataset

| Property | Value |
|---|---|
| File | `heart.csv` |
| Rows | 918 |
| Features | 11 |
| Target | `HeartDisease` (0 = no disease, 1 = disease) |
| Class balance | ~55% positive, ~45% negative |

### Features

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Age of the patient in years |
| Sex | Categorical | M = male, F = female |
| ChestPainType | Categorical | ATA, ASY, NAP, or TA |
| RestingBP | Numeric | Resting blood pressure (mm Hg) |
| Cholesterol | Numeric | Serum cholesterol (mm/dl) |
| FastingBS | Categorical | Fasting blood sugar > 120 mg/dl (1 = true) |
| RestingECG | Categorical | Normal, ST, or LVH |
| MaxHR | Numeric | Maximum heart rate achieved |
| ExerciseAngina | Categorical | Exercise-induced angina (Y/N) |
| Oldpeak | Numeric | ST depression induced by exercise |
| ST_Slope | Categorical | Slope of the peak exercise ST segment (Up, Flat, Down) |

---

## Project Structure

```
├── heart.csv
├── heart_disease_ml.py     # Main notebook (run in Jupyter)
└── README.md
```

---

## Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

Python 3.8 or above is recommended.

---

## How to Run

1. Place `heart.csv` and `heart_disease_ml.py` in the same folder.
2. Open Jupyter Notebook or JupyterLab.
3. Upload or open the `.py` file — the `# %%` cell markers are recognised natively by Jupyter and VS Code.
4. Run all cells top to bottom (`Kernel → Restart & Run All`).

---

## Notebook Walkthrough

### 1. Imports & Load Dataset
Loads `heart.csv` with pandas and prints shape and a preview of the first rows.

### 2. Exploratory Data Analysis
- Checks for missing values (there are none)
- Prints class balance of the target variable
- Plots target distribution and age histogram split by class

### 3. Preprocessing
- **Label encoding** for binary categoricals: `Sex` (M→1, F→0) and `ExerciseAngina` (Y→1, N→0)
- **One-hot encoding** for multi-class columns: `ChestPainType`, `RestingECG`, `ST_Slope`
- **Train/test split**: 80% train, 20% test, stratified to preserve class ratio
- **StandardScaler**: fit on training data only, then applied to test — used for Linear and Logistic Regression only

### 4. Decision Tree

**4a — max_depth tuning**
Loops through depths 1–20, recording train and test accuracy at each step. Plots both curves to visualise the overfitting crossover point. The best depth is selected automatically based on test accuracy.

**4b — Cross-validation**
Runs a `StratifiedKFold(n_splits=5)` cross-validation with the best depth. Stratified folds preserve the class ratio in every split. Reports per-fold scores, mean, and standard deviation.

**4c — Final model**
Trains the tuned Decision Tree and produces a classification report, a visual tree plot (top 3 levels), feature importance bar chart, and confusion matrix.

### 5. Linear Regression (Baseline)
Fits a standard linear regressor on the scaled data and thresholds predictions at 0.5 to produce binary class labels. Used as a performance floor — predictions outside [0, 1] are its key limitation. Includes a coefficient plot showing which features push the prediction up or down.

### 6. Logistic Regression

**6a — GridSearchCV hyperparameter tuning**
Searches 15 values of the regularisation parameter `C` on a log scale from 10⁻⁵ to 10⁸ using 5-fold cross-validation. Plots mean CV accuracy vs. C with a shaded confidence band.

> Tuned result: `C = 0.006105` — Best CV score: `0.853`

The very small best C indicates strong regularisation was needed, likely due to multicollinearity introduced by one-hot encoded columns.

**6b — Final model**
Trains logistic regression with the best C, then produces a classification report, odds ratio bar chart (exponentiated coefficients), and an ROC curve.

### 7. Naive Bayes
Trains `GaussianNB` — no scaling needed. Assumes numeric features follow a Gaussian distribution per class. Produces a classification report and confusion matrix.

### 8. Model Comparison Summary
Collects accuracy and AUC-ROC for all four models, prints a ranked table, and plots side-by-side bar charts and overlaid ROC curves.

---

## Results Summary

| Model | Accuracy (approx.) | AUC-ROC | Scaling needed |
|---|---|---|---|
| Logistic Regression | ~87% | ~0.93 | Yes |
| Decision Tree | ~83% | ~0.88 | No |
| Naive Bayes | ~82% | ~0.88 | No |
| Linear Regression | ~80% | ~0.87 | Yes |

---

## Key Observations

1. **ST_Slope, ChestPainType, and Oldpeak** are the most important features — consistent with their clinical significance as ECG and symptom indicators of cardiac stress.
2. **Logistic Regression performs best** overall, benefiting from regularisation tuned via GridSearchCV. The best C of 0.006105 with a CV score of 0.853 shows the model needed strong regularisation to generalise.
3. The **Decision Tree is the most interpretable** model — its rules can be read directly by a clinician, which often matters more than a 2–3% accuracy gap in a medical setting.
4. **Naive Bayes is competitive** despite its independence assumption, confirming that the individual features carry enough signal about heart disease risk on their own.
5. For clinical applications, **Recall for the Disease class should be prioritised** over raw accuracy — a missed diagnosis (false negative) is more costly than a false alarm (false positive).

---

## Metrics Used

| Metric | Why it matters here |
|---|---|
| Accuracy | Overall % correct — reliable here since classes are near-balanced |
| Precision | Of predicted disease cases, how many were actually disease |
| Recall | Of actual disease cases, how many did we catch — critical in medical diagnosis |
| F1 Score | Harmonic mean of precision and recall — good single summary |
| AUC-ROC | Threshold-independent ranking quality across all decision boundaries |
