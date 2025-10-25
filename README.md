# Telecom Customer Churn Prediction

The goal is to **predict customer churn** in a telecommunications company using **machine learning models**, and to evaluate the solution from both **technical** and **business** perspectives.

---

## Objective

Develop a **supervised classification model** to estimate the probability of customer churn and enable the company to prioritize **retention actions**.  
Both **technical metrics** (ROC-AUC, PR-AUC, F1, Recall) and **business metrics** (Top-N, ROI proxy) were implemented.

---

## Dataset

- **Source:** IBM Telco Customer Churn  
- **Rows:** 7,043  
- **Columns:** 21  
- **Target variable:** `Churn (Yes/No)`

---

## Methodology

**1. EDA & Cleaning**  
- Handling missing values, data types, and category unification.

**2. Preprocessing**  
- Pipelines including imputation, scaling, categorical encoding, and business-driven feature engineering (`services_count`, `tenure_band`, etc.).

**3. Modeling**  
- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- SVM  
- Naive Bayes  

**4. Evaluation**  
- Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC.

**5. Interpretability**  
- SHAP values.

---

## Results

**Final Model:** Logistic Regression  

**Key Metrics:**
- ROC-AUC ≈ **0.862**  
- Precision ≈ **0.687**  
- Recall ≈ **0.600**  
- F1 ≈ **0.641**

---

## ⚙️ Execution

### 🔹 Local Environment

```bash
pip install -r requirements.txt
jupyter notebook notebooks/Proyecto_final_grupo_H.ipynb
