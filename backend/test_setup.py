import pandas as pd

df = pd.read_csv("data/student.csv")

print(df.head())
print("\nShape:", df.shape)
print("\nDataset loaded successfully ✅")
print(df.columns)