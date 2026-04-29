
import pandas as pd
import numpy as np
from pathlib import Path
from deep_clv_model import DeepCLVModel
from sklearn.preprocessing import StandardScaler

# Load data
DATA_FILE = Path(__file__).resolve().parent.parent / "dataset" / "Online Retail.xlsx"
df = pd.read_excel(DATA_FILE)
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# RFM calculation
snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})
rfm.columns = ['Recency','Frequency','Monetary']
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']].values)
y = rfm['Monetary_log'].values

# Train and save DL model
model = DeepCLVModel()
model.fit(X, y, epochs=100, batch_size=32)
model.save("models/deep_clv_model.keras")
import joblib
joblib.dump(scaler, "models/deep_clv_scaler.pkl")
print("Deep Learning CLV model trained and saved to models/deep_clv_model.keras")
