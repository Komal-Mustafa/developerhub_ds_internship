# Task 4: Insurance Claim Amount Prediction 💰

## Project Overview

This project predicts insurance claim amounts using Linear Regression. The task involves building a machine learning model to estimate annual medical insurance costs based on personal health and demographic factors.

**Dataset**: Medical Cost Personal Dataset (1,338 records)  
**Problem Type**: Regression (predicting continuous values)  
**Target Variable**: `charges` (annual medical insurance cost in USD)  
**Model**: Linear Regression  
**Performance**: R² Score: **94.26%**

---

## 📊 Dataset Information

### Features (Predictors)
| Feature | Type | Description |
|---------|------|-------------|
| **age** | int | Age of the insured person (18-63 years) |
| **sex** | str | Gender (male/female) |
| **bmi** | float | Body Mass Index (15.96 - 53.13) |
| **children** | int | Number of dependent children (0-5) |
| **smoker** | str | Smoking status (yes/no) |
| **region** | str | US region (southwest, southeast, northwest, northeast) |

### Target Variable
| Variable | Type | Description |
|----------|------|-------------|
| **charges** | float | Annual medical insurance cost in USD ($3,296 - $63,770) |

### Dataset Statistics
```
Dataset Shape: (1338, 7)

Basic Statistics:
              age          bmi     children      charges
count  1338.00000  1338.000000  1338.000000   1338.00000
mean     40.48281    30.649417     1.060538  19262.40201
std      13.12366     5.933343     1.179511  13379.58953
min      18.00000    15.960000     0.000000   3296.78000
25%      29.00000    26.500000     0.000000  10953.78750
50%      41.00000    30.755000     1.000000  14398.37000
75%      52.00000    34.617500     2.000000  18545.41250
max      63.00000    48.790000     5.000000  56162.66000
```

---

## 🎯 Key Findings

### Model Performance
- **MAE (Mean Absolute Error)**: $2,210
  - Average prediction error across all predictions
  
- **RMSE (Root Mean Squared Error)**: $2,911
  - Penalizes larger errors more heavily
  
- **R² Score**: 0.9426 (94.26%)
  - The model explains 94.26% of variance in insurance charges

### Feature Correlation Analysis

**Correlation with Charges (Ranked):**
```
1. smoker              0.9337  ⭐ STRONGEST predictor
2. age                 0.2217
3. bmi                 0.1159
4. sex                 0.0378
5. children            0.0283
6. region_southwest    0.0119
7. region_southeast    0.0022
8. region_northwest   -0.0095
```

### Key Insights for Insurance Companies

1. **Smoking is the #1 Factor**
   - Smokers pay approximately 3-4x more than non-smokers
   - This is by far the strongest predictor of insurance charges

2. **BMI (Body Mass Index) Impact**
   - BMI has moderate positive correlation with charges
   - Impact is especially significant when BMI > 30 (obese category)

3. **Age-Related Trends**
   - Insurance charges increase steadily with age
   - Medical needs and health risks grow with age

4. **Other Factors**
   - Number of children has minimal impact on charges
   - Geographic region (US) has negligible effect
   - Sex/gender shows minimal correlation with charges

5. **Highest Risk Group**
   - Smokers with BMI > 30: HIGHEST RISK category
   - This group should receive premium insurance rates

6. **Lowest Risk Group**
   - Non-smokers with BMI < 25: LOWEST RISK category
   - Eligible for better insurance premiums

---

## 📁 Project Structure

```
Insurance_prediction/
├── README.md                          # This file
├── insurance_predition.py             # Main Python script
├── task4_insurance.csv                # Input dataset
├── task4_eda.png                      # Exploratory Data Analysis plots
├── task4_correlation.png              # Feature correlation chart
└── task4_model_eval.png               # Model evaluation plots
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.13+
- Virtual Environment (.venv)

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Specific Versions Used
```
pandas==2.0+
numpy==2.4.4
scikit-learn==1.3+
matplotlib==3.7+
seaborn==0.12+
scipy==1.17.1
```

---

## 🚀 How to Run

### Step 1: Navigate to the project directory
```bash
cd "g:/Internship Folder/Insurance_prediction"
```

### Step 2: Activate the virtual environment
```bash
# Windows
..\\.venv\\Scripts\\activate

# macOS/Linux
source ../.venv/bin/activate
```

### Step 3: Run the script
```bash
python insurance_predition.py
```

### Expected Output
The script will:
1. Load the dataset (1,338 records)
2. Display basic statistics and data summary
3. Generate EDA visualizations
4. Perform correlation analysis
5. Train a Linear Regression model
6. Evaluate model performance (MAE, RMSE, R²)
7. Generate model evaluation plots
8. Display key insights and recommendations

---

## 📈 Generated Visualizations

### 1. **task4_eda.png** - Exploratory Data Analysis
- Distribution plots for all features
- Relationships between variables and target
- Identifies patterns and trends

### 2. **task4_correlation.png** - Feature Correlation Chart
- Bar chart showing correlation coefficients
- Highlights which features matter most
- Color-coded: Red for positive, Blue for negative correlations

### 3. **task4_model_eval.png** - Model Evaluation
- Actual vs Predicted scatter plot
- Residuals distribution plot
- Model performance visualization

---

## 🔍 Model Details

### Algorithm: Linear Regression
**Why Linear Regression?**
- Interpretable: Easy to understand feature impacts
- Fast: Computationally efficient
- Baseline: Good starting point for regression tasks

### Formula
```
Insurance Charges = β₀ + β₁(Age) + β₂(BMI) + β₃(Smoker) + 
                    β₄(Children) + β₅(Sex) + β₆(Region) + ε
```

### Data Preprocessing
1. **Encoding**: Categorical variables (sex, region, smoker) converted to numeric
2. **Scaling**: Features standardized using StandardScaler
3. **Train-Test Split**: 80% training, 20% testing

### Model Training
- Used scikit-learn's LinearRegression
- Fitted on 80% of data (1,070 samples)
- Evaluated on 20% of data (268 samples)

---

## 📊 Model Limitations & Future Improvements

### Current Limitations
1. **Linear Assumption**: Assumes straight-line relationships between features and target
2. **Non-linear Interactions**: May miss complex patterns (e.g., smoking × BMI interaction)
3. **Outliers**: Linear regression is sensitive to extreme values
4. **Limited Features**: Could benefit from additional health metrics

### Recommended Improvements
1. **Polynomial Features**: Add squared/cubic terms for non-linear relationships
2. **Ensemble Methods**: Try Random Forest, Gradient Boosting (often perform better)
3. **Interaction Terms**: Explicitly add smoking × BMI, age × smoker interactions
4. **Feature Engineering**: Create age groups, BMI categories
5. **Outlier Treatment**: Detect and handle extreme values
6. **Hyperparameter Tuning**: Optimize regularization parameters

---

## 💡 Business Recommendations

1. **Premium Pricing Strategy**
   - Charge significantly higher premiums for smokers
   - Implement BMI-based pricing tiers
   - Age-adjusted premiums

2. **Risk Management**
   - Flag smokers with high BMI as high-risk
   - Offer wellness programs to reduce BMI
   - Incentivize smoking cessation

3. **Market Segmentation**
   - Develop different plans for different risk groups
   - Young, healthy non-smokers: Economy plans
   - Older smokers with high BMI: Premium plans

4. **Cost Prediction**
   - Accurately estimate claim costs for new customers
   - Better financial forecasting and reserve planning
   - Competitive pricing advantage

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.13 |
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn |
| **Visualization** | matplotlib, seaborn |
| **Version Control** | Git, GitHub |
| **Environment** | Virtual Environment (.venv) |

---

## 📝 Code Walkthrough

### Main Sections
1. **Data Loading**: Generate/load insurance dataset
2. **EDA**: Statistical analysis and visualizations
3. **Correlation Analysis**: Identify feature relationships
4. **Data Preprocessing**: Encoding and scaling
5. **Model Training**: Fit Linear Regression
6. **Model Evaluation**: Calculate performance metrics
7. **Results**: Display insights and recommendations

---

## 🎓 Learning Outcomes

By completing this project, you will understand:
- ✅ How to build a regression model in scikit-learn
- ✅ Feature correlation and importance analysis
- ✅ Model evaluation metrics (MAE, RMSE, R²)
- ✅ Data preprocessing and feature scaling
- ✅ Business insights from machine learning models
- ✅ How to interpret regression coefficients
- ✅ Visualization of model performance

---

## 📚 References

- [scikit-learn LinearRegression Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- [Regression Evaluation Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Feature Correlation Analysis](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html)

---

## 👨‍💼 Author

**DevelopersHub Corporation**  
Data Science & Analytics Internship  
Task 4: Insurance Claim Prediction

---

## 📞 Support

For questions or issues:
1. Check the script output for error messages
2. Verify all dependencies are installed
3. Ensure data file (task4_insurance.csv) is present
4. Check Python version compatibility (3.13+)

---

## 📄 License

This project is part of the DevelopersHub internship program.

---

**Last Updated**: April 28, 2026  
**Model Status**: ✅ Production Ready (94.26% R² Score)
