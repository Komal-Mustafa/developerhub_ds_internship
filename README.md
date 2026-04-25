# 📊 DevelopersHub Data Science & Analytics Internship
### Intern: Komal  Mustafa DHC_321
 |

---

## 🗂️ Repository Structure

```
developershub-ds-internship/
│
├── task1_iris_exploration/
│   ├── task1_iris_exploration.py
│   ├── README.md
│   └── outputs/  (scatter_plots.png, histograms.png, box_plots.png, etc.)
│
├── task2_credit_risk/
│   ├── task2_credit_risk.py
│   ├── README.md
│   └── outputs/  (eda.png, confusion_matrix.png)
│
├── task3_churn_prediction/
│   ├── task3_churn_prediction.py
│   ├── README.md
│   └── outputs/  (eda.png, feature_importance.png)
│
├── task4_insurance_prediction/
│   ├── task4_insurance_prediction.py
│   ├── README.md
│   └── outputs/  (eda.png, correlation.png, model_eval.png)
│
├── task5_loan_acceptance/
│   ├── task5_loan_acceptance.py
│   ├── README.md
│   └── outputs/  (eda.png, confusion_matrix.png)
│
└── README.md  ← (You are here)
```

---

## ✅ Tasks Completed

| Task | Title | Status | Model Used | Accuracy |
|------|-------|--------|-----------|---------|
| Task 1 | Iris Dataset Exploration & Visualization | ✅ Done | N/A (EDA only) | — |
| Task 2 | Credit Risk Prediction | ✅ Done | Logistic Regression + Decision Tree | ~82% |
| Task 3 | Customer Churn Prediction | ✅ Done | Random Forest + Logistic Regression | ~85% |
| Task 4 | Insurance Claim Prediction | ✅ Done | Linear Regression | R²~0.85 |
| Task 5 | Personal Loan Acceptance | ✅ Done | Logistic Regression + Decision Tree | ~88% |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Primary language |
| pandas | Data loading, cleaning, manipulation |
| numpy | Numerical operations |
| matplotlib | Base plotting |
| seaborn | Statistical visualizations |
| scikit-learn | ML models, preprocessing, evaluation |

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/developershub-ds-internship.git
cd developershub-ds-internship

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# Run any task
python task1_iris_exploration/task1_iris_exploration.py
```

---

## 📚 Datasets Used

All datasets are either built-in (seaborn) or synthetically generated to match real-world Kaggle datasets of the same structure:

| Task | Dataset | Real-World Source |
|------|---------|------------------|
| Task 1 | Iris Dataset | seaborn built-in / UCI ML Repo |
| Task 2 | Loan Prediction Dataset | 
| Task 3 | Churn Modelling Dataset |
| Task 4 | Medical Cost Personal | 
| Task 5 | Bank Personal Loan | 

---

## 🔑 Key Learnings

1. **Data Cleaning** — Real datasets always have missing values; strategy matters (mean vs mode vs median)
2. **EDA First** — Always visualize before modeling; plots reveal patterns models confirm
3. **Feature Engineering** — Creating new features (e.g., Total_Income) often boosts performance
4. **Encoding** — Label encoding for binary categories; one-hot for multi-class categories
5. **Model Evaluation** — Accuracy alone isn't enough; always check confusion matrix + classification report
6. **Business Insights** — The goal isn't just accuracy — it's finding actionable business recommendations

---
