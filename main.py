import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report



#  Load the CSV
df = pd.read_csv("telco_customer_churn.csv")

#  Data cleaning
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df.dropna(subset=["TotalCharges"], inplace=True)

#   EDA
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x="Contract", hue="Churn", data=df)
plt.title("Churn by Contract Type")
plt.savefig("churn_by_contract_type.png")
plt.show()

sns.countplot(x="InternetService", hue="Churn", data=df)
plt.title("Churn by Internet Service")
plt.savefig("churn_by_internet_service.png")
plt.show()

sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, kde=True)
plt.title("Monthly Charges by Churn")
plt.savefig("monthly_charges_by_churn.png")
plt.show()


#  One-hot encode categorical variables
df_encoded = pd.get_dummies(df, drop_first=True)


#  Define features and target
X = df_encoded.drop("Churn_Yes", axis=1)
y = df_encoded["Churn_Yes"]

#  Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

plt.figure(figsize=(12, 8))
sns.heatmap(df_encoded.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()



# Train logistic regression model
model = LogisticRegression(class_weight="balanced", max_iter=1000)
model.fit(X_train, y_train)


model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("✅ Model training complete!")
print("🔍 Accuracy:", accuracy_score(y_test, y_pred))
print("\n🧾 Classification Report:\n", classification_report(y_test, y_pred))


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Plot top 10 features most associated with churn (Logistic Regression coefficients)
coefficients = pd.Series(model.coef_[0], index=X.columns)
coefficients.sort_values().tail(10).plot(kind='barh')
plt.title("Top 10 Positive Churn Indicators")
plt.xlabel("Coefficient")
plt.tight_layout()
plt.savefig("top_10_positive_churn_indicators.png")
plt.show()
