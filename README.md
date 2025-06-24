# Credit Risk Prediction

This project uses machine learning models to predict credit risk based on customer financial and demographic data. It includes two predictive tasks:

1. **Classification Task**: Predict whether a customer will have default records (`Defaults Records` = 0 or ≥1).
2. **Regression Task**: Predict a customer's `Credit Score`.

---

## 📁 Dataset

- **File**: `risk_behavior_features.csv`
- **Rows**: 74
- **Columns**: 12
- **Features Include**:
  - Age
  - Gender
  - Education Level
  - Marital Status
  - Number of Dependents
  - Income
  - Credit Score
  - Debt-to-Income Ratio
  - Assets Value
  - Defaults Records
  - Employment Status
  - Years in Current Job

---

## 🧪 Models Used

- **Logistic Regression** – for binary classification of default risk.
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Random Forest Regressor** – for credit score prediction.

---

## 🧰 Workflow

1. **Data Cleaning**
   - Handle missing values
   - Drop duplicates

2. **Exploratory Data Analysis (EDA)**
   - Distribution plots
   - Correlation heatmap
   - Boxplots by default status

3. **Feature Engineering**
   - Encoding categorical variables
   - Creating binary target for classification

4. **Preprocessing**
   - Train-test split
   - Feature scaling

5. **Modeling**
   - Logistic Regression
   - Random Forest
   - XGBoost

6. **Evaluation**
   - Confusion Matrix
   - Classification Report
   - RMSE, R² for regression

---

## 📊 Results

### Classification (Defaults Records)
- **Accuracy**: 100%
- **Precision, Recall, F1**: All 1.00 (likely due to small dataset; cross-validation is recommended)

### Regression (Credit Score)
- **RMSE**: ~47
- **R² Score**: ~0.46

---

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/Mohaemnsaber/credit-risk-prediction.git
   cd credit-risk-prediction
