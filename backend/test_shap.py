import pandas as pd
from shap_utils import explain_prediction, get_feature_importance

# Load sample data
df = pd.read_csv("model/processed_data.csv")
X = df.drop("risk", axis=1)

# Take one sample
sample = X.iloc[[0]]

# Get SHAP values
shap_vals = explain_prediction(sample)
print("SHAP values for one prediction:")
print(shap_vals)

# Global feature importance
print("\nTop Features:")
importance = get_feature_importance()
for k, v in list(importance.items())[:5]:
    print(k, ":", round(v, 3))