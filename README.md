# 📊 Machine Learning Projects – Assignment by Badkul Technologies

This repository contains two complete machine learning projects that I worked on as part of an **assignment given by Badkul Technologies**.  
Both projects are based on popular Kaggle datasets and showcase the **end-to-end machine learning pipeline**: data preprocessing, feature engineering, model training, evaluation, and final submission.  

---

## 📌 Introduction

Machine Learning is not just about building models – it is about preparing the data, understanding the problem, and evaluating models properly.  
In this assignment, I worked on two classic datasets:  

1. **House Price Prediction 🏠** → A regression problem where the task is to predict the final sale price of houses.  
2. **Titanic Survival Prediction 🛳️** → A classification problem where the goal is to predict whether a passenger survived or not.  

Both datasets are widely used for practicing **data preprocessing, feature engineering, and model selection**.  

---

## 🎯 Objectives

- Learn and implement **data preprocessing techniques** such as handling missing values, scaling, and encoding categorical features.  
- Perform **feature engineering** to improve model performance.  
- Train and compare multiple ML models (linear, tree-based, and boosting).  
- Use **cross-validation** to evaluate model performance fairly.  
- Generate a **final Kaggle-compatible submission file** for each project.  

---

## 🏠 Project 1: House Price Prediction

- **Dataset:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
- **Type:** Regression  

### 🔹 Methodology
1. **Data Loading** → Loaded `train.csv` and `test.csv`.  
2. **Data Preprocessing**  
   - Handled missing values:
     - Numerical → filled with **median**  
     - Categorical → filled with **"Missing"**  
   - Scaled numerical features using `StandardScaler`.  
   - Encoded categorical variables using `OneHotEncoder`.  
3. **Feature Engineering**  
   - Applied **log transformation** on the target variable (`SalePrice`) to reduce skewness.  
4. **Model Building**  
   - Ridge Regression (linear)  
   - Random Forest Regressor (non-linear ensemble)  
   - XGBoost Regressor (boosting, if installed)  
5. **Model Evaluation**  
   - Used **5-fold cross-validation**.  
   - Metric: **Root Mean Squared Error (RMSE)**.  
6. **Final Submission**  
   - Best model (Random Forest) was retrained on the **entire training dataset**.  
   - Predictions generated for test set and saved in `submission_house.csv`.  

---

## 🛳️ Project 2: Titanic Survival Prediction

- **Dataset:** [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
- **Type:** Classification  

### 🔹 Methodology
1. **Data Loading** → Loaded `train.csv` and `test.csv`.  
2. **Data Preprocessing**  
   - Filled missing values (`Age`, `Embarked`, `Cabin`).  
   - Converted categorical variables like `Sex` and `Embarked` into numerical form.  
3. **Feature Engineering**  
   - Created `FamilySize` = `SibSp + Parch + 1`.  
   - Created `IsAlone` feature (1 if no family members, else 0).  
   - Extracted passenger **Title** from names.  
4. **Model Building**  
   - Logistic Regression (baseline linear model).  
   - Random Forest Classifier (ensemble).  
   - Support Vector Machine (non-linear).  
5. **Model Evaluation**  
   - Used cross-validation with **Accuracy** and **F1-score**.  
6. **Final Submission**  
   - Best model retrained on the **entire training dataset**.  
   - Predictions generated for test set and saved in `submission_titanic.csv`.  

---

## 📂 Repository Structure
# 📊 Machine Learning Projects – Assignment by Badkul Technologies

This repository contains two complete machine learning projects that I worked on as part of an **assignment given by Badkul Technologies**.  
Both projects are based on popular Kaggle datasets and showcase the **end-to-end machine learning pipeline**: data preprocessing, feature engineering, model training, evaluation, and final submission.  

---

## 📌 Introduction

Machine Learning is not just about building models – it is about preparing the data, understanding the problem, and evaluating models properly.  
In this assignment, I worked on two classic datasets:  

1. **House Price Prediction 🏠** → A regression problem where the task is to predict the final sale price of houses.  
2. **Titanic Survival Prediction 🛳️** → A classification problem where the goal is to predict whether a passenger survived or not.  

Both datasets are widely used for practicing **data preprocessing, feature engineering, and model selection**.  

---

## 🎯 Objectives

- Learn and implement **data preprocessing techniques** such as handling missing values, scaling, and encoding categorical features.  
- Perform **feature engineering** to improve model performance.  
- Train and compare multiple ML models (linear, tree-based, and boosting).  
- Use **cross-validation** to evaluate model performance fairly.  
- Generate a **final Kaggle-compatible submission file** for each project.  

---

## 🏠 Project 1: House Price Prediction

- **Dataset:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
- **Type:** Regression  

### 🔹 Methodology
1. **Data Loading** → Loaded `train.csv` and `test.csv`.  
2. **Data Preprocessing**  
   - Handled missing values:
     - Numerical → filled with **median**  
     - Categorical → filled with **"Missing"**  
   - Scaled numerical features using `StandardScaler`.  
   - Encoded categorical variables using `OneHotEncoder`.  
3. **Feature Engineering**  
   - Applied **log transformation** on the target variable (`SalePrice`) to reduce skewness.  
4. **Model Building**  
   - Ridge Regression (linear)  
   - Random Forest Regressor (non-linear ensemble)  
   - XGBoost Regressor (boosting, if installed)  
5. **Model Evaluation**  
   - Used **5-fold cross-validation**.  
   - Metric: **Root Mean Squared Error (RMSE)**.  
6. **Final Submission**  
   - Best model (Random Forest) was retrained on the **entire training dataset**.  
   - Predictions generated for test set and saved in `submission_house.csv`.  

---

## 🛳️ Project 2: Titanic Survival Prediction

- **Dataset:** [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
- **Type:** Classification  

### 🔹 Methodology
1. **Data Loading** → Loaded `train.csv` and `test.csv`.  
2. **Data Preprocessing**  
   - Filled missing values (`Age`, `Embarked`, `Cabin`).  
   - Converted categorical variables like `Sex` and `Embarked` into numerical form.  
3. **Feature Engineering**  
   - Created `FamilySize` = `SibSp + Parch + 1`.  
   - Created `IsAlone` feature (1 if no family members, else 0).  
   - Extracted passenger **Title** from names.  
4. **Model Building**  
   - Logistic Regression (baseline linear model).  
   - Random Forest Classifier (ensemble).  
   - Support Vector Machine (non-linear).  
5. **Model Evaluation**  
   - Used cross-validation with **Accuracy** and **F1-score**.  
6. **Final Submission**  
   - Best model retrained on the **entire training dataset**.  
   - Predictions generated for test set and saved in `submission_titanic.csv`.  

---

## 📂 Repository Structure
# 📊 Machine Learning Projects – Assignment by Badkul Technologies

This repository contains two complete machine learning projects that I worked on as part of an **assignment given by Badkul Technologies**.  
Both projects are based on popular Kaggle datasets and showcase the **end-to-end machine learning pipeline**: data preprocessing, feature engineering, model training, evaluation, and final submission.  

---

## 📌 Introduction

Machine Learning is not just about building models – it is about preparing the data, understanding the problem, and evaluating models properly.  
In this assignment, I worked on two classic datasets:  

1. **House Price Prediction 🏠** → A regression problem where the task is to predict the final sale price of houses.  
2. **Titanic Survival Prediction 🛳️** → A classification problem where the goal is to predict whether a passenger survived or not.  

Both datasets are widely used for practicing **data preprocessing, feature engineering, and model selection**.  

---

## 🎯 Objectives

- Learn and implement **data preprocessing techniques** such as handling missing values, scaling, and encoding categorical features.  
- Perform **feature engineering** to improve model performance.  
- Train and compare multiple ML models (linear, tree-based, and boosting).  
- Use **cross-validation** to evaluate model performance fairly.  
- Generate a **final Kaggle-compatible submission file** for each project.  

---

## 🏠 Project 1: House Price Prediction

- **Dataset:** [Kaggle – House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)  
- **Type:** Regression  

### 🔹 Methodology
1. **Data Loading** → Loaded `train.csv` and `test.csv`.  
2. **Data Preprocessing**  
   - Handled missing values:
     - Numerical → filled with **median**  
     - Categorical → filled with **"Missing"**  
   - Scaled numerical features using `StandardScaler`.  
   - Encoded categorical variables using `OneHotEncoder`.  
3. **Feature Engineering**  
   - Applied **log transformation** on the target variable (`SalePrice`) to reduce skewness.  
4. **Model Building**  
   - Ridge Regression (linear)  
   - Random Forest Regressor (non-linear ensemble)  
   - XGBoost Regressor (boosting, if installed)  
5. **Model Evaluation**  
   - Used **5-fold cross-validation**.  
   - Metric: **Root Mean Squared Error (RMSE)**.  
6. **Final Submission**  
   - Best model (Random Forest) was retrained on the **entire training dataset**.  
   - Predictions generated for test set and saved in `submission_house.csv`.  

---

## 🛳️ Project 2: Titanic Survival Prediction

- **Dataset:** [Kaggle – Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic)  
- **Type:** Classification  

### 🔹 Methodology
1. **Data Loading** → Loaded `train.csv` and `test.csv`.  
2. **Data Preprocessing**  
   - Filled missing values (`Age`, `Embarked`, `Cabin`).  
   - Converted categorical variables like `Sex` and `Embarked` into numerical form.  
3. **Feature Engineering**  
   - Created `FamilySize` = `SibSp + Parch + 1`.  
   - Created `IsAlone` feature (1 if no family members, else 0).  
   - Extracted passenger **Title** from names.  
4. **Model Building**  
   - Logistic Regression (baseline linear model).  
   - Random Forest Classifier (ensemble).  
   - Support Vector Machine (non-linear).  
5. **Model Evaluation**  
   - Used cross-validation with **Accuracy** and **F1-score**.  
6. **Final Submission**  
   - Best model retrained on the **entire training dataset**.  
   - Predictions generated for test set and saved in `submission_titanic.csv`.  

---

## 📂 Repository Structure

```bash
├── HousePrices/
│   ├── train.csv
│   ├── test.csv
│   ├── house_prices.ipynb
│   └── submission_house.csv
│
├── Titanic/
│   ├── train.csv
│   ├── test.csv
│   ├── titanic.ipynb
│   └── submission_titanic.csv
│
└── README.md

---

## ⚙️ Tech Stack

- **Programming Language:** Python (3.8+)  
- **Libraries Used:**
  - **Pandas, NumPy** → Data cleaning & manipulation  
  - **Matplotlib, Seaborn** → Data visualization  
  - **Scikit-learn** → Preprocessing, Ridge, Random Forest, Logistic Regression, SVM  
  - **XGBoost** (optional, for house price boosting model)  

---

## 📊 Results & Insights

- For **House Prices**, Random Forest performed the best (XGBoost not installed), achieving the lowest RMSE.  
- For **Titanic Survival**, Random Forest gave the highest accuracy, outperforming Logistic Regression and SVM.  
- **Key Insight:**  
  - Linear models are simple but often underperform on complex datasets.  
  - Tree-based models (Random Forest, XGBoost) handle non-linear relationships better.  

---

## ✅ Conclusion

These projects demonstrate the **complete workflow of a Machine Learning task**:  
- Understanding the dataset  
- Cleaning and preprocessing  
- Feature engineering  
- Training multiple models  
- Evaluating fairly with cross-validation  
- Generating a proper Kaggle submission  

Through this assignment, I strengthened my knowledge of **machine learning pipelines, preprocessing, model selection, and evaluation techniques**.  

---

## 📝 Acknowledgment

This repository is part of an **assignment given by Badkul Technologies**.  
Working on these projects helped me gain **hands-on practical experience** with:  
- Data preprocessing pipelines  
- Handling missing data  
- Encoding categorical features  
- Model building and evaluation  
- Preparing Kaggle submissions  

---



