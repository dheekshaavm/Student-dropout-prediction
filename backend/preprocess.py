import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
df = pd.read_csv("data/student.csv", sep=";")

# Create target column (dropout risk)
# G3 < 10 = At risk
df["risk"] = (df["G3"] < 10).astype(int)

# Drop final grade (since we used it as target)
df = df.drop("G3", axis=1)

# Encode categorical columns
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Create model folder if not exists
os.makedirs("model", exist_ok=True)

# Save processed data
df.to_csv("model/processed_data.csv", index=False)

print("Preprocessing complete ✅")
print("Saved to: model/processed_data.csv")
print("Shape:", df.shape)
