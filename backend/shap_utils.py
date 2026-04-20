import shap
import joblib
import pandas as pd

# Load trained model
model = joblib.load("model/model.pkl")

# Create SHAP explainer (Tree-based models like XGBoost)
explainer = shap.TreeExplainer(model)

def explain_prediction(input_df):
    """
    Returns SHAP values for given input dataframe
    """
    shap_values = explainer.shap_values(input_df)
    return shap_values

def get_feature_importance():
    """
    Global feature importance using SHAP
    """
    # Load training data for global explanation
    df = pd.read_csv("model/processed_data.csv")
    X = df.drop("risk", axis=1)

    shap_values = explainer.shap_values(X)
    importance = abs(shap_values).mean(axis=0)

    feature_importance = dict(zip(X.columns, importance))
    return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))