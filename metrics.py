from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


ROOT_DIR = Path(__file__).resolve().parent
DATA_FILE = ROOT_DIR / "dataset" / "Online Retail.xlsx"
STATIC_DIR = ROOT_DIR / "static"


def load_rfm_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_excel(path)
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

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


def summarize_scores(name: str, mae_list: list, rmse_list: list, r2_list: list) -> dict:
    return {
        "Model": name,
        "MAE Mean": float(np.mean(mae_list)),
        "MAE Std": float(np.std(mae_list)),
        "RMSE Mean": float(np.mean(rmse_list)),
        "RMSE Std": float(np.std(rmse_list)),
        "R2 Mean": float(np.mean(r2_list)),
        "R2 Std": float(np.std(r2_list)),
    }


def evaluate_deep_model(X: np.ndarray, y_log: np.ndarray, kf: KFold) -> dict:
    mae_scores, rmse_scores, r2_scores = [], [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_log[train_idx], y_log[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(3,)),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, verbose=0)

        y_pred_log = model.predict(X_test_scaled, verbose=0).flatten()
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_test)

        mae_scores.append(mean_absolute_error(y_true, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2_scores.append(r2_score(y_true, y_pred))

    return summarize_scores("Deep Learning (DNN)", mae_scores, rmse_scores, r2_scores)


def evaluate_baselines(X_df: pd.DataFrame, y_log: pd.Series, kf: KFold) -> list:
    models = [
        ("Linear Regression", LinearRegression()),
        ("Ridge Regression", Ridge()),
        ("Lasso Regression", Lasso()),
        ("Decision Tree", DecisionTreeRegressor(random_state=42)),
        ("Random Forest", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
        ("KNN Regression", KNeighborsRegressor(n_neighbors=5, weights="distance")),
    ]

    outputs = []
    for name, model in models:
        mae_scores, rmse_scores, r2_scores = [], [], []
        for train_idx, test_idx in kf.split(X_df):
            X_train, X_test = X_df.iloc[train_idx], X_df.iloc[test_idx]
            y_train, y_test = y_log.iloc[train_idx], y_log.iloc[test_idx]

            model.fit(X_train, y_train)
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            y_true = np.expm1(y_test)

            mae_scores.append(mean_absolute_error(y_true, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_true, y_pred)))
            r2_scores.append(r2_score(y_true, y_pred))

        outputs.append(summarize_scores(name, mae_scores, rmse_scores, r2_scores))

    return outputs


def save_comparison_plot(results_df: pd.DataFrame) -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)

    sorted_df = results_df.sort_values("R2 Mean", ascending=False)
    x = np.arange(len(sorted_df))

    plt.figure(figsize=(12, 6))
    plt.bar(x - 0.25, sorted_df["MAE Mean"], width=0.25, label="MAE")
    plt.bar(x, sorted_df["RMSE Mean"], width=0.25, label="RMSE")
    plt.bar(x + 0.25, sorted_df["R2 Mean"], width=0.25, label="R2")
    plt.xticks(x, sorted_df["Model"], rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("CLV Model Comparison (5-Fold CV)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(STATIC_DIR / "model_comparison.png")
    plt.close()


if __name__ == "__main__":
    print("Loading and preparing RFM features...")
    rfm = load_rfm_data(DATA_FILE)

    X_df = rfm[["Recency", "Frequency", "Monetary"]]
    y_log = np.log1p(rfm["Monetary"])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    print("Evaluating deep learning model...")
    deep_result = evaluate_deep_model(X_df.values, y_log.values, kf)

    print("Evaluating baseline ML models...")
    baseline_results = evaluate_baselines(X_df, y_log, kf)

    all_results = [deep_result] + baseline_results
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values("R2 Mean", ascending=False)

    print("\n--- Regression Metrics (CLV, 5-fold CV) ---")
    for _, row in results_df.iterrows():
        print(
            f"{row['Model']:22} | "
            f"MAE: {row['MAE Mean']:.2f} ± {row['MAE Std']:.2f} | "
            f"RMSE: {row['RMSE Mean']:.2f} ± {row['RMSE Std']:.2f} | "
            f"R2: {row['R2 Mean']:.3f} ± {row['R2 Std']:.3f}"
        )

    save_comparison_plot(results_df)
    print("\nSaved comparison chart to static/model_comparison.png")
