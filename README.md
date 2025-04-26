# Heart Disease Prediction Using Machine Learning

## Overview
This project applies three machine learning models — **Logistic Regression**, **Random Forest**, and **XGBoost** — to predict the presence and severity of heart disease.  
The goal is to compare the models’ performance and evaluate which one is most effective for this dataset.

---

## Dataset
- **Source:** [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Samples:** 303 patients
- **Features:** 13 clinical attributes + 1 target (`num`, values 0–4)
- **Target:** Predict severity of heart disease (5-class classification)

---

## Methods
- Data cleaning (handling missing values)
- Feature scaling (for Logistic Regression)
- Train/Test Split (80/20)
- Model training:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- Visualization:
  - Confusion matrices
  - Accuracy comparison bar plot

---

## Results

| Model                | Accuracy | Precision | Recall | F1-Score |
|:---------------------|:--------:|:---------:|:------:|:--------:|
| Logistic Regression  | 61.7%    | 34.3%     | 34.6%  | 33.8%    |
| Random Forest        | 56.7%    | 26.5%     | 26.9%  | 26.1%    |
| XGBoost              | 56.7%    | 25.0%     | 27.1%  | 25.8%    |

---

## Key Takeaways
- Logistic Regression and Random Forest performed similarly on this dataset.
- XGBoost slightly outperformed the other models.
- Real-world clinical datasets are often small and noisy, making model performance highly dependent on data quality.

---

## Future Improvements
- Hyperparameter tuning (e.g., using GridSearchCV)
- Cross-validation for robustness
- Feature selection/engineering
- Experiment with ensemble learning methods

---

## Requirements
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - matplotlib
  - seaborn

---

## Project Structure
heart-disease-ml/
│
├── heart_disease_prediction.ipynb   # Full Jupyter notebook
├── README.md                        # Project overview
├── requirements.txt                 # (Optional) Library list
└── images/                          # (Optional) Saved plots

---

## ✍️ Author
- Riley Coleman  
---
