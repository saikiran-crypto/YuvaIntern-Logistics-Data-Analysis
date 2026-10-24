import pandas as pd

data = {
    "Shipment_ID": [101, 102, 103, 104, 105],
    "Delivery_Days": [2, 4, 3, 6, 2],
    "Promised_Days": [3, 3, 3, 4, 3],
    "Transport_Cost": [500, 750, 600, 900, 450]
}

df = pd.DataFrame(data)

on_time_rate = (
    (df["Delivery_Days"] <= df["Promised_Days"]).mean() * 100
)

delay_rate = (
    (df["Delivery_Days"] > df["Promised_Days"]).mean() * 100
)

average_delivery_time = df["Delivery_Days"].mean()
average_transport_cost = df["Transport_Cost"].mean()
shipment_volume = len(df)

print(f"On-time delivery rate: {on_time_rate:.1f}%")
print(f"Delivery delay rate: {delay_rate:.1f}%")
print(f"Average delivery time: {average_delivery_time:.2f} days")
print(f"Average transport cost: {average_transport_cost:.2f}")
print(f"Shipment volume: {shipment_volume}")