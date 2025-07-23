# Telco Customer Churn Prediction

This project uses machine learning to analyze and predict customer churn using the Telco Customer Churn dataset from Kaggle.

---

# Objective

Predict whether a customer will churn based on features like contract type, tenure, monthly charges, and internet service.

---

# Dataset

- **File:** `telco_customer_churn.csv`  
- **Source:** [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

# Tools & Libraries

- **Programming:** Python  
- **Libraries:** pandas, numpy, seaborn, matplotlib, scikit-learn  
- **Environment:** VS Code / Jupyter Notebook

---

# Exploratory Data Analysis (EDA) Insights

- Churn is higher among customers with month-to-month contracts.
- Customers with fiber optic internet and higher monthly charges are more likely to churn.
- Shorter customer tenure is strongly associated with higher churn rates.

---



---

# Model Summary

- **Model:** Logistic Regression  
- **Class Imbalance:** Handled with `class_weight="balanced"`  
- **Accuracy:** ~78.5%  
- **Recall (Churners):** ~51%

---

# Project Structure

| File                      | Description                                |
|---------------------------|--------------------------------------------|
| `main.py`                 | Full Python script (cleaning, EDA, modeling) |
| `telco_customer_churn.csv`| Dataset used for training                  |
| `README.md`               | Project summary and visuals                |
| `.png files`              | Saved graphs and visualizations            |

---

# Future Improvements

- Try other models (e.g., Random Forest, XGBoost)
- Perform hyperparameter tuning
- Deploy as a web app using Streamlit or Flask

---

# Author

**Jazmin Piva**  
[Connect on LinkedIn](https://www.linkedin.com/in/jazmin-piva/)


