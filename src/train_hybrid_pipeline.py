import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from deep_clv_model import DeepCLVModel


def load_and_prepare_data(data_file: Path) -> pd.DataFrame:
    df = pd.read_excel(data_file)
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("CustomerID").agg(
        {
            "InvoiceDate": lambda x: (snapshot_date - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum",
        }
    )
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    return rfm


def compute_segment_map(kmeans_model: KMeans) -> dict:
    centers = kmeans_model.cluster_centers_
    n_clusters = centers.shape[0]

    high_value_votes = [centers[:, 0].argmin(), centers[:, 1].argmax(), centers[:, 2].argmax()]
    high_value_idx = max(set(high_value_votes), key=high_value_votes.count)

    lost_votes = [centers[:, 0].argmax(), centers[:, 1].argmin(), centers[:, 2].argmin()]
    lost_idx = max(set(lost_votes), key=lost_votes.count)

    remaining = list(set(range(n_clusters)) - {high_value_idx, lost_idx})
    remaining.sort(key=lambda idx: centers[idx, 1], reverse=True)

    segment_map = {
        high_value_idx: "High Value Customer",
        lost_idx: "Lost Customer",
    }

    if len(remaining) >= 1:
        segment_map[remaining[0]] = "Regular Customer"
    if len(remaining) >= 2:
        segment_map[remaining[1]] = "Occasional Customer"
    for idx in remaining[2:]:
        segment_map[idx] = "Occasional Customer"

    return segment_map


def assign_rule_segment(recency: float, frequency: float, monetary: float, thresholds: dict) -> str:
    if recency <= thresholds["recency"] and frequency > thresholds["frequency"] and monetary > thresholds["monetary"]:
        return "High Value Customer"
    if recency > thresholds["recency"]:
        return "Lost Customer"
    if frequency > thresholds["frequency"]:
        return "Regular Customer"
    return "Occasional Customer"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train integrated ML + DL customer intelligence pipeline.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for deep CLV model.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for deep CLV model.")
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parent.parent
    data_file = root_dir / "dataset" / "Online Retail.xlsx"
    model_dir = root_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Loading and preprocessing dataset...")
    df = load_and_prepare_data(data_file)
    rfm = build_rfm(df)

    thresholds = {
        "recency": float(rfm["Recency"].quantile(0.75)),
        "frequency": float(rfm["Frequency"].quantile(0.50)),
        "monetary": float(rfm["Monetary"].quantile(0.50)),
    }

    print("Training KMeans segmentation model...")
    cluster_scaler = StandardScaler()
    rfm_scaled = cluster_scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(rfm_scaled)
    segment_map = compute_segment_map(kmeans)

    print("Training classical CLV model (Linear Regression)...")
    ml_clv_model = LinearRegression()
    ml_clv_model.fit(rfm[["Recency", "Frequency", "Monetary"]], rfm["Monetary"])

    print("Training deep CLV model (Dense Neural Network)...")
    deep_scaler = StandardScaler()
    X_dl = deep_scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]].values)
    y_dl = np.log1p(rfm["Monetary"].values)
    deep_model = DeepCLVModel()
    deep_model.fit(X_dl, y_dl, epochs=args.epochs, batch_size=args.batch_size)

    print("Saving artifacts for Flask inference pipeline...")
    joblib.dump(cluster_scaler, model_dir / "scaler.pkl")
    joblib.dump(kmeans, model_dir / "kmeans_model.pkl")
    joblib.dump(segment_map, model_dir / "segment_map.pkl")
    joblib.dump(thresholds, model_dir / "rfm_thresholds.pkl")
    joblib.dump(ml_clv_model, model_dir / "clv_model.pkl")
    joblib.dump(deep_scaler, model_dir / "deep_clv_scaler.pkl")
    deep_model.save(str(model_dir / "deep_clv_model.keras"))

    print("\nArtifacts generated:")
    print("- models/scaler.pkl")
    print("- models/kmeans_model.pkl")
    print("- models/segment_map.pkl")
    print("- models/rfm_thresholds.pkl")
    print("- models/clv_model.pkl")
    print("- models/deep_clv_scaler.pkl")
    print("- models/deep_clv_model.keras")

    demo_row = rfm.iloc[0]
    recency, frequency, monetary = float(demo_row["Recency"]), float(demo_row["Frequency"]), float(demo_row["Monetary"])
    rule_segment = assign_rule_segment(recency, frequency, monetary, thresholds)
    cluster_segment = segment_map[int(clusters[0])]

    ml_clv = float(ml_clv_model.predict([[recency, frequency, monetary]])[0])
    ml_clv = max(0.0, ml_clv)
    deep_input = deep_scaler.transform([[recency, frequency, monetary]])
    deep_clv = float(deep_model.predict(deep_input)[0])

    print("\nDemo prediction on first customer record:")
    print(f"- RFM: Recency={recency:.1f}, Frequency={frequency:.1f}, Monetary={monetary:.2f}")
    print(f"- Rule-based segment: {rule_segment}")
    print(f"- KMeans cluster segment: {cluster_segment}")
    print(f"- ML CLV prediction: ${ml_clv:.2f}")
    print(f"- DL CLV prediction: ${deep_clv:.2f}")
    print("\nHybrid pipeline training completed successfully.")


if __name__ == "__main__":
    main()
