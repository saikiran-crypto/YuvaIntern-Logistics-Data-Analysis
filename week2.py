import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Illustrative sample records only; not real company data.
data = {
    "Shipment_ID": [101, 102, 103, 104, 104, 105, 106, 107],
    "Delivery_Days": [2, 4, None, 6, 6, 3, 45, 5],
    "Promised_Days": [3, 3, 4, 4, 4, 4, 5, 5],
    "Transport_Cost": [500, 750, 600, None, None, 450, 12000, 800],
    "Shipment_Weight": [10, 25, 15, 20, 20, None, 30, 18],
    "Delivery_Status": ["On Time", "Late", "On Time", "Late", "Late", "On Time", "Late", "On Time"]
}
df = pd.DataFrame(data)

print("Raw shape:", df.shape)
print("\nMissing values before cleaning:\n", df.isnull().sum())
print("\nDuplicate rows before cleaning:", df.duplicated().sum())

# Remove exact duplicate rows.
df = df.drop_duplicates()

# Convert numeric fields and handle missing values using medians.
numeric_cols = ["Delivery_Days", "Promised_Days", "Transport_Cost", "Shipment_Weight"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

# Detect potential outliers using IQR and cap values at the IQR fences.
outlier_cols = ["Delivery_Days", "Transport_Cost", "Shipment_Weight"]
for col in outlier_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower=lower, upper=upper)

# Normalize selected numeric columns to a 0-1 range.
scaler = MinMaxScaler()
for col in ["Delivery_Days", "Transport_Cost", "Shipment_Weight"]:
    df[col + "_Normalized"] = scaler.fit_transform(df[[col]])

# Calculate delay days (never negative).
df["Delay_Days"] = (df["Delivery_Days"] - df["Promised_Days"]).clip(lower=0)

df.to_csv("logistics_cleaned_week2.csv", index=False)
print("\nCleaned shape:", df.shape)
print("\nMissing values after cleaning:\n", df.isnull().sum())
print("\nCleaned preview:\n", df.head())
print("\nSaved: logistics_cleaned_week2.csv")