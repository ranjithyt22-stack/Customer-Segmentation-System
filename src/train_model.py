import pandas as pd
import datetime as dt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import joblib
import os

# Create models folder if not exists
os.makedirs("models", exist_ok=True)

print("Loading dataset...")

df = pd.read_excel("dataset/Online Retail.xlsx")

# Data cleaning
df = df.dropna(subset=['CustomerID'])
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Create total price
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Snapshot date
snapshot_date = df['InvoiceDate'].max() + dt.timedelta(days=1)

# Create RFM table
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})

rfm.columns = ['Recency','Frequency','Monetary']

print("RFM Features Created")


# Derive percentile thresholds for rule-based segmentation
r_threshold = rfm['Recency'].quantile(0.75)
f_threshold = rfm['Frequency'].quantile(0.50)
m_threshold = rfm['Monetary'].quantile(0.50)

thresholds = {
    "recency": float(r_threshold),
    "frequency": float(f_threshold),
    "monetary": float(m_threshold)
}


def assign_segment(row):
    if (
        row['Recency'] <= thresholds['recency']
        and row['Frequency'] > thresholds['frequency']
        and row['Monetary'] > thresholds['monetary']
    ):
        return "High Value Customer"
    if row['Recency'] > thresholds['recency']:
        return "Lost Customer"
    if row['Frequency'] > thresholds['frequency']:
        return "Regular Customer"
    return "Occasional Customer"


rfm['Segment'] = rfm.apply(assign_segment, axis=1)


# Normalize features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency','Frequency','Monetary']])

# Elbow method plot
inertia = []
K_range = range(1, 10)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)
plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.title('Elbow Method For Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.tight_layout()
plt.savefig('static/elbow_plot.png')
plt.close()

# Train KMeans
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print("KMeans model trained")

# Analyze cluster centers
cluster_summary = rfm.groupby('Cluster').mean()


# Print cluster centers for inspection
print("\nCluster Centers (RFM order):")
for idx, center in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {idx}: Recency={center[0]:.2f}, Frequency={center[1]:.2f}, Monetary={center[2]:.2f}")

print("\nCluster Summary:")
print(cluster_summary)

# Determine segment meaning

# Robust segment mapping based on RFM values
segment_map = {}
centers = kmeans.cluster_centers_

# High Value: low recency, high frequency, high monetary
high_value = ((centers[:,0].argmin(), centers[:,1].argmax(), centers[:,2].argmax()))
high_value_idx = max(set(high_value), key=high_value.count)

# Lost: high recency, low frequency, low monetary
lost = ((centers[:,0].argmax(), centers[:,1].argmin(), centers[:,2].argmin()))
lost_idx = max(set(lost), key=lost.count)

# Remove used indices
remaining = set(range(4)) - {high_value_idx, lost_idx}
remaining = list(remaining)

# Assign remaining clusters by frequency/monetary
if centers[remaining[0],1] > centers[remaining[1],1]:
    regular_idx, occasional_idx = remaining[0], remaining[1]
else:
    regular_idx, occasional_idx = remaining[1], remaining[0]

segment_map[high_value_idx] = "High Value Customer"
segment_map[regular_idx] = "Regular Customer"
segment_map[occasional_idx] = "Occasional Customer"
segment_map[lost_idx] = "Lost Customer"

print("\nSegment Mapping (robust):")
print(segment_map)


# Train CLV regression model (predict future value from RFM)
clv_model = LinearRegression()
clv_model.fit(rfm[['Recency', 'Frequency', 'Monetary']], rfm['Monetary'])
joblib.dump(clv_model, "models/clv_model.pkl")

# Persist thresholds for inference-time segmentation
joblib.dump(thresholds, "models/rfm_thresholds.pkl")

# Save model
joblib.dump(kmeans, "models/kmeans_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(segment_map, "models/segment_map.pkl")

print("\nModel, mappings, and CLV regressor saved successfully!")