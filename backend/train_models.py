import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Load processed dataset
df = pd.read_csv("model/processed_data.csv")

# Split features and target
X = df.drop("risk", axis=1)
y = df["risk"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training models...\n")

# ============================
# 1️⃣ Logistic Regression
# ============================
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_acc = accuracy_score(y_test, lr_pred)

print("Logistic Regression Accuracy:", lr_acc)

# ============================
# 2️⃣ Random Forest
# ============================
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)

# ============================
# 3️⃣ XGBoost (Primary Model)
# ============================
xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_acc = accuracy_score(y_test, xgb_pred)

print("XGBoost Accuracy:", xgb_acc)

# ============================
# Save best model (XGBoost)
# ============================
os.makedirs("model", exist_ok=True)
joblib.dump(xgb, "model/model.pkl")

print("\nBest model saved as model/model.pkl ✅")

# Optional: print detailed report
print("\nClassification Report (XGBoost):")
print(classification_report(y_test, xgb_pred))