import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Create illustrative sample logistics data
data = {
    "Delivery_Days": [
        2, 5, 3, 6, 4, 2, 5, 3, 7, 4,
        1, 6, 3, 5, 2, 7, 4, 3, 6, 2
    ],
    "Promised_Days": [
        3, 4, 3, 4, 4, 3, 4, 3, 5, 4,
        2, 5, 4, 4, 3, 5, 5, 3, 5, 2
    ],
    "Transport_Cost": [
        450, 780, 520, 950, 700, 430, 820, 560, 1100, 690,
        300, 990, 610, 800, 480, 1150, 720, 590, 1020, 410
    ],
    "Shipment_Weight_kg": [
        8, 15, 10, 22, 18, 7, 16, 11, 25, 14,
        5, 24, 12, 17, 9, 26, 19, 13, 23, 6
    ],
    "Distance_km": [
        50, 220, 80, 350, 180, 45, 240, 95, 420, 160,
        25, 380, 110, 260, 60, 450, 200, 100, 400, 40
    ]
}

df = pd.DataFrame(data)

# 2. Create target variable
df["Delayed"] = (
    df["Delivery_Days"] > df["Promised_Days"]
).astype(int)

# 3. Select features and target
features = [
    "Promised_Days",
    "Transport_Cost",
    "Shipment_Weight_kg",
    "Distance_km"
]

X = df[features]
y = df["Delayed"]

# 4. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# 5. Train the Decision Tree model
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# 6. Predict test results
y_pred = model.predict(X_test)

# 7. Evaluate the model
print("=" * 45)
print("     LOGISTICS DELAY PREDICTION")
print("=" * 45)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nModel Accuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    zero_division=0
))

# 8. Predict delays for all shipments
df["Predicted_Delayed"] = model.predict(X)

# 9. Calculate median transportation cost
median_cost = df["Transport_Cost"].median()

# 10. Assign review priorities
def assign_priority(row):
    if (
        row["Predicted_Delayed"] == 1
        and row["Transport_Cost"] >= median_cost
    ):
        return "High"
    elif row["Predicted_Delayed"] == 1:
        return "Medium"
    else:
        return "Normal"


df["Review_Priority"] = df.apply(
    assign_priority,
    axis=1
)

# 11. Display shipment predictions
print("\n" + "=" * 45)
print("          SHIPMENT PREDICTIONS")
print("=" * 45)

print(
    df[
        [
            "Delivery_Days",
            "Promised_Days",
            "Transport_Cost",
            "Predicted_Delayed",
            "Review_Priority"
        ]
    ].to_string(index=False)
)

# 12. Create output folder
output_folder = Path("week4_outputs")
output_folder.mkdir(exist_ok=True)

# 13. Save prediction results
output_file = output_folder / "logistics_week4_predictions.csv"

df.to_csv(output_file, index=False)

print("\nPredictions saved successfully!")
print("Output file:", output_file.resolve())

print("\nNote: This program uses illustrative sample data.")