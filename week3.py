
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Illustrative sample shipment records only
data = {
    "Shipment_ID": [201, 202, 203, 204, 205, 206, 207, 208, 209, 210],
    "Region": ["North", "South", "East", "West", "North", "South", "East", "West", "North", "South"],
    "Delivery_Days": [2, 5, 3, 6, 4, 2, 5, 3, 7, 4],
    "Promised_Days": [3, 4, 3, 4, 4, 3, 4, 3, 5, 4],
    "Transport_Cost": [450, 780, 520, 950, 700, 430, 820, 560, 1100, 690],
    "Shipment_Weight_kg": [8, 15, 10, 22, 18, 7, 16, 11, 25, 14],
}
df = pd.DataFrame(data)

df["Delay_Days"] = (df["Delivery_Days"] - df["Promised_Days"]).clip(lower=0)
df["On_Time"] = df["Delivery_Days"] <= df["Promised_Days"]

print("=== Logistics Data Overview ===")
print(df.head())
print(f"\nTotal shipments: {len(df)}")
print(f"Average delivery time: {df['Delivery_Days'].mean():.2f} days")
print(f"Average transport cost: {df['Transport_Cost'].mean():.2f}")
print(f"On-time delivery rate: {df['On_Time'].mean() * 100:.1f}%")
print(f"Delayed shipments: {(~df['On_Time']).sum()}")

print("\n=== Region Summary ===")
region_summary = df.groupby("Region").agg(
    Shipments=("Shipment_ID", "count"),
    Average_Delivery_Days=("Delivery_Days", "mean"),
    Average_Transport_Cost=("Transport_Cost", "mean"),
    On_Time_Rate=("On_Time", "mean"),
).round(2)
print(region_summary)

out_dir = Path("week3_outputs")
out_dir.mkdir(exist_ok=True)

# 1. Delivery time by shipment
plt.figure(figsize=(8, 5))
plt.bar(df["Shipment_ID"].astype(str), df["Delivery_Days"])
plt.title("Delivery Time by Shipment")
plt.xlabel("Shipment ID")
plt.ylabel("Delivery Days")
plt.tight_layout()
plt.savefig(out_dir / "delivery_time_by_shipment.png", dpi=150)
plt.close()

# 2. Average transportation cost by region
cost_by_region = df.groupby("Region")["Transport_Cost"].mean().sort_index()
plt.figure(figsize=(8, 5))
plt.bar(cost_by_region.index, cost_by_region.values)
plt.title("Average Transportation Cost by Region")
plt.xlabel("Region")
plt.ylabel("Average Cost")
plt.tight_layout()
plt.savefig(out_dir / "average_cost_by_region.png", dpi=150)
plt.close()

# 3. Delivery days versus transportation cost
plt.figure(figsize=(8, 5))
plt.scatter(df["Delivery_Days"], df["Transport_Cost"])
plt.title("Delivery Time vs Transportation Cost")
plt.xlabel("Delivery Days")
plt.ylabel("Transportation Cost")
plt.tight_layout()
plt.savefig(out_dir / "delivery_vs_cost.png", dpi=150)
plt.close()

# Save analysis outputs
df.to_csv(out_dir / "logistics_week3_analyzed.csv", index=False)
region_summary.to_csv(out_dir / "region_summary.csv")

print(f"\nSaved charts and CSV outputs in: {out_dir.resolve()}")
print("Reminder: findings describe only the illustrative sample data.")