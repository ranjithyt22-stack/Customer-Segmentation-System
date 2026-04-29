
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import joblib
import pandas as pd
import os
import io
import base64
import importlib
from src.visualization import generate_dashboard_plots
from src.interactive_dashboard import generate_interactive_dashboard
from sklearn.metrics import silhouette_score


app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load both CLV model variants when available so users can choose at runtime.
DEEP_CLV_MODEL_PATH = "models/deep_clv_model.keras"
DEEP_CLV_SCALER_PATH = "models/deep_clv_scaler.pkl"
CUSTOMER_LOOKUP_CACHE_PATH = "models/customer_lookup.pkl"
RFM_DASHBOARD_CACHE_PATH = "models/rfm_dashboard_cache.pkl"

deep_clv_available = os.path.exists(DEEP_CLV_MODEL_PATH) and os.path.exists(DEEP_CLV_SCALER_PATH)
ml_clv_available = True
deep_clv_model = None
deep_clv_scaler = None
ml_clv_model = None

if deep_clv_available:
    try:
        deep_clv_scaler = joblib.load(DEEP_CLV_SCALER_PATH)
    except FileNotFoundError:
        deep_clv_available = False
        print("Deep CLV artifacts missing. Deep model option will be unavailable.")
else:
    print("Deep CLV artifacts missing. Deep model option will be unavailable.")

try:
    ml_clv_model = joblib.load("models/clv_model.pkl")
except FileNotFoundError:
    ml_clv_available = False
    print("ML CLV artifact missing. ML model option will be unavailable.")


def refresh_deep_model_availability():
    global deep_clv_available, deep_clv_scaler
    model_exists = os.path.exists(DEEP_CLV_MODEL_PATH)
    scaler_exists = os.path.exists(DEEP_CLV_SCALER_PATH)
    deep_clv_available = model_exists and scaler_exists
    if deep_clv_available and deep_clv_scaler is None:
        try:
            deep_clv_scaler = joblib.load(DEEP_CLV_SCALER_PATH)
        except FileNotFoundError:
            deep_clv_available = False
            print("Deep CLV scaler missing after refresh. Deep model option will be unavailable.")
    return deep_clv_available


def ensure_deep_model_loaded():
    global deep_clv_model, deep_clv_available
    if not refresh_deep_model_availability():
        return False
    if deep_clv_model is not None:
        return True
    try:
        deep_module = importlib.import_module("src.deep_clv_model")
        deep_clv_model = deep_module.DeepCLVModel.load(DEEP_CLV_MODEL_PATH)
        return True
    except Exception as exc:
        deep_clv_available = False
        print(f"Deep CLV model failed to load: {exc}")
        return False

# --- RFM and Segment Caching ---
rfm_cache = None
segment_percentages_cache = None
total_customers_cache = None


def warm_dashboard_cache(force_refresh=False):
    global rfm_cache, segment_percentages_cache, total_customers_cache

    if rfm_cache is None or force_refresh:
        rfm = load_rfm_dashboard_cache(force_refresh=force_refresh)
        if rfm is None:
            return False
        rfm_cache = rfm
        total_customers_cache = len(rfm)
        segment_counts = rfm['Segment'].value_counts()
        segment_percentages_cache = {
            seg: round(100 * count / total_customers_cache, 1)
            for seg, count in segment_counts.items()
        }

    # Generate heavy assets once unless explicitly forced
    needs_assets = (
        force_refresh
        or not os.path.exists("static/cluster_plot.png")
        or not os.path.exists("static/segment_pie.png")
        or not os.path.exists("static/avg_spending.png")
        or not os.path.exists("static/cluster_plot_interactive.html")
        or not os.path.exists("static/segment_pie_interactive.html")
        or not os.path.exists("static/avg_spending_interactive.html")
    )
    if needs_assets and rfm_cache is not None:
        generate_dashboard_plots(rfm_cache)
        generate_interactive_dashboard(rfm_cache)

    return rfm_cache is not None


def has_interactive_dashboard_assets():
    return (
        os.path.exists("static/cluster_plot_interactive.html")
        and os.path.exists("static/segment_pie_interactive.html")
        and os.path.exists("static/avg_spending_interactive.html")
    )

def get_rfm():
    global rfm_cache
    if rfm_cache is None:
        warm_dashboard_cache(force_refresh=False)
    return rfm_cache

# Strategy recommendations
strategies = {
    "High Value Customer": "Provide loyalty rewards and premium offers.",
    "Regular Customer": "Offer cross-selling and product bundles.",
    "Occasional Customer": "Send promotional discounts.",
    "Lost Customer": "Send re-engagement campaigns and special discounts."
}

segment_offer_badges = {
    "High Value Customer": "Gold Member: 20% off Premium Silk Collection",
    "Regular Customer": "Bundle Bonus: Buy 2 Cotton Sets, Get 10% Off",
    "Occasional Customer": "Welcome Back: 12% off Any Fabric This Week",
    "Lost Customer": "Comeback Coupon: Flat 15% Off on First Return Bill",
}

segment_fabric_preferences = {
    "High Value Customer": "Silk",
    "Regular Customer": "Cotton",
    "Occasional Customer": "Polyester Blends",
    "Lost Customer": "Seasonal Mixed Fabrics",
}

segment_visuals = {
    "High Value Customer": {
        "label": "High Value",
        "theme": "ci-segment-high",
        "icon": "bi-gem",
    },
    "Regular Customer": {
        "label": "New Customer",
        "theme": "ci-segment-new",
        "icon": "bi-gift",
    },
    "Occasional Customer": {
        "label": "At Risk",
        "theme": "ci-segment-risk",
        "icon": "bi-hourglass-split",
    },
    "Lost Customer": {
        "label": "Lost",
        "theme": "ci-segment-lost",
        "icon": "bi-archive",
    },
}

USD_TO_INR = 83  # Approximate conversion for display purposes


def build_offer_payload(segment, customer_id):
    coupon_map = {
        "High Value Customer": {
            "title": "Exclusive 20% Off",
            "subtitle": "Premium Silk Collection",
            "savings_inr": 1200,
        },
        "Regular Customer": {
            "title": "Fresh Arrival Bonus",
            "subtitle": "15% off on Cotton and Daily Wear",
            "savings_inr": 700,
        },
        "Occasional Customer": {
            "title": "We Miss You Offer",
            "subtitle": "Flat INR 500 off on your next purchase",
            "savings_inr": 500,
        },
        "Lost Customer": {
            "title": "Comeback Coupon",
            "subtitle": "Flat INR 500 off on your next purchase",
            "savings_inr": 500,
        },
    }
    selected = coupon_map.get(
        segment,
        {
            "title": "Special Offer",
            "subtitle": "Ask cashier for in-store discount",
            "savings_inr": 300,
        },
    )
    segment_code = "".join([s[0] for s in segment.split()][:3]).upper()
    customer_part = str(customer_id).replace(" ", "").replace("-", "")[-4:] or "0000"
    coupon_code = f"{segment_code}{customer_part}26"
    return {
        "title": selected["title"],
        "subtitle": selected["subtitle"],
        "savings_inr": selected["savings_inr"],
        "coupon_code": coupon_code,
    }


def build_offer_qr_data_uri(coupon_code):
    qr_payload = f"TEXTILE-OFFER:{coupon_code}"
    try:
        qrcode_module = importlib.import_module("qrcode")
    except ModuleNotFoundError:
        return ""
    qr = qrcode_module.QRCode(version=1, box_size=4, border=1)
    qr.add_data(qr_payload)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def normalize_rfm_scores(recency, frequency, monetary):
    recency_score = max(0, min(100, round((180 - recency) / 180 * 100)))
    frequency_target = max(thresholds["frequency"] * 2, 1)
    monetary_target = max(thresholds["monetary"] * 2, 1)
    frequency_score = max(0, min(100, round((frequency / frequency_target) * 100)))
    monetary_score = max(0, min(100, round((monetary / monetary_target) * 100)))
    return {
        "recency": recency_score,
        "frequency": frequency_score,
        "monetary": monetary_score,
    }


def load_clean_dataset():
    try:
        df = pd.read_excel("dataset/Online Retail.xlsx")
    except FileNotFoundError:
        return None
    df = df.dropna(subset=["CustomerID"])
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df


def compute_rfm(df):
    snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    })
    rfm.columns = ['Recency','Frequency','Monetary']
    return rfm


def compute_thresholds_from_rfm(rfm):
    return {
        "recency": float(rfm['Recency'].quantile(0.75)),
        "frequency": float(rfm['Frequency'].quantile(0.50)),
        "monetary": float(rfm['Monetary'].quantile(0.50)),
    }


def load_thresholds():
    try:
        return joblib.load("models/rfm_thresholds.pkl")
    except FileNotFoundError:
        df = load_clean_dataset()
        if df is None:
            raise FileNotFoundError("Customer dataset not found and thresholds file missing.")
        rfm = compute_rfm(df)
        thresholds = compute_thresholds_from_rfm(rfm)
        os.makedirs("models", exist_ok=True)
        joblib.dump(thresholds, "models/rfm_thresholds.pkl")
        return thresholds


def load_customer_lookup():
    dataset_path = "dataset/Online Retail.xlsx"
    if os.path.exists(CUSTOMER_LOOKUP_CACHE_PATH) and os.path.exists(dataset_path):
        cache_is_fresh = os.path.getmtime(CUSTOMER_LOOKUP_CACHE_PATH) >= os.path.getmtime(dataset_path)
        if cache_is_fresh:
            try:
                return joblib.load(CUSTOMER_LOOKUP_CACHE_PATH)
            except Exception as exc:
                print(f"Customer lookup cache could not be loaded and will be rebuilt: {exc}")

    df = load_clean_dataset()
    if df is None:
        print("Customer dataset not found. Fetch endpoint will be unavailable.")
        return None
    summary = df.groupby("CustomerID").agg(
        last_purchase=("InvoiceDate", "max"),
        total_purchases=("InvoiceNo", "nunique"),
        total_spending=("TotalPrice", "sum"),
    )
    summary["last_purchase"] = summary["last_purchase"].dt.normalize()
    os.makedirs("models", exist_ok=True)
    joblib.dump(summary, CUSTOMER_LOOKUP_CACHE_PATH)
    return summary


def load_rfm_dashboard_cache(force_refresh=False):
    dataset_path = "dataset/Online Retail.xlsx"
    if (
        not force_refresh
        and os.path.exists(RFM_DASHBOARD_CACHE_PATH)
        and os.path.exists(dataset_path)
        and os.path.getmtime(RFM_DASHBOARD_CACHE_PATH) >= os.path.getmtime(dataset_path)
    ):
        try:
            return joblib.load(RFM_DASHBOARD_CACHE_PATH)
        except Exception as exc:
            print(f"RFM dashboard cache could not be loaded and will be rebuilt: {exc}")

    customer_summary = load_customer_lookup()
    if customer_summary is not None:
        snapshot_date = customer_summary["last_purchase"].max() + pd.Timedelta(days=1)
        rfm = pd.DataFrame(index=customer_summary.index)
        rfm["Recency"] = (snapshot_date - customer_summary["last_purchase"]).dt.days
        rfm["Frequency"] = customer_summary["total_purchases"]
        rfm["Monetary"] = customer_summary["total_spending"]
    else:
        df = load_clean_dataset()
        if df is None:
            return None
        rfm = compute_rfm(df)

    rfm["Segment"] = rfm.apply(
        lambda row: assign_segment(row["Recency"], row["Frequency"], row["Monetary"]),
        axis=1
    )
    os.makedirs("models", exist_ok=True)
    joblib.dump(rfm, RFM_DASHBOARD_CACHE_PATH)
    return rfm


customer_lookup = None
customer_lookup_by_id = None
dataset_reference_date = pd.Timestamp.today().normalize()
thresholds = load_thresholds()


def ensure_customer_lookup_loaded():
    global customer_lookup, customer_lookup_by_id, dataset_reference_date
    if customer_lookup is None:
        customer_lookup = load_customer_lookup()
        if customer_lookup is not None:
            dataset_reference_date = customer_lookup["last_purchase"].max() + pd.Timedelta(days=1)
            customer_lookup_by_id = {
                normalize_customer_id(customer_id): record
                for customer_id, record in customer_lookup.iterrows()
            }
    return customer_lookup


def normalize_customer_id(value):
    raw = str(value).strip()
    if not raw:
        return None
    try:
        numeric = float(raw)
    except ValueError:
        return None
    if numeric.is_integer():
        return str(int(numeric))
    return str(numeric)


def assign_segment(recency, frequency, monetary):
    if (
        recency <= thresholds["recency"]
        and frequency > thresholds["frequency"]
        and monetary > thresholds["monetary"]
    ):
        return "High Value Customer"
    if recency > thresholds["recency"]:
        return "Lost Customer"
    if frequency > thresholds["frequency"]:
        return "Regular Customer"
    return "Occasional Customer"


@app.route("/")
def home():
    refresh_deep_model_availability()
    return render_template(
        "index.html",
        model_selected="auto",
        deep_clv_available=deep_clv_available,
        ml_clv_available=ml_clv_available,
        prediction_count=session.get("prediction_count", 0)
    )



@app.route("/predict", methods=["POST"])
def predict():
    # Automatic RFM calculation from user-friendly input
    customer_id = request.form["customer_id"]
    last_purchase = request.form["last_purchase"]
    total_purchases = float(request.form["total_purchases"])
    total_spending = float(request.form["total_spending"])
    last_purchase_date = pd.to_datetime(last_purchase)
    if ensure_customer_lookup_loaded() is not None:
        reference_date = dataset_reference_date
    else:
        reference_date = pd.Timestamp.today().normalize()
    recency = max((reference_date - last_purchase_date).days, 0)
    frequency = total_purchases
    monetary = total_spending
    segment = assign_segment(recency, frequency, monetary)
    strategy = strategies.get(segment, "No strategy available.")
    offer_badge = segment_offer_badges.get(segment, "Special in-store offer available")
    preferred_fabric = segment_fabric_preferences.get(segment, "General Fabrics")
    rfm_scores = normalize_rfm_scores(recency, frequency, monetary)
    segment_ui = segment_visuals.get(
        segment,
        {"label": segment, "theme": "ci-segment-new", "icon": "bi-person"},
    )
    offer = build_offer_payload(segment, customer_id)
    offer_qr_data = build_offer_qr_data_uri(offer["coupon_code"])
    model_selected = request.form.get("model_choice", "auto")
    prediction_count = session.get("prediction_count", 0) + 1
    session["prediction_count"] = prediction_count

    if model_selected == "deep":
        if not ensure_deep_model_loaded() or deep_clv_model is None:
            flash("Deep Learning model is not loaded. Please retrain or check model files.")
            return render_template(
                "index.html",
                model_selected=model_selected,
                deep_clv_available=deep_clv_available,
                ml_clv_available=ml_clv_available,
                result=None,
                error_message="Deep Learning model is not loaded. Please retrain or check model files.",
                prediction_count=prediction_count
            )
        try:
            X_input = deep_clv_scaler.transform([[recency, frequency, monetary]])
            clv_usd = float(deep_clv_model.predict(X_input)[0])
            model_used = "Deep Learning"
        except Exception as exc:
            import traceback
            tb = traceback.format_exc()
            print("Deep model prediction error:\n", tb)
            flash(f"Deep Learning model error: {exc}")
            return render_template(
                "index.html",
                model_selected=model_selected,
                deep_clv_available=deep_clv_available,
                ml_clv_available=ml_clv_available,
                result=None,
                error_message=f"Deep Learning model error: {exc}",
                prediction_count=prediction_count
            )
    elif model_selected == "ml" and ml_clv_available:
        clv_usd = float(ml_clv_model.predict([[recency, frequency, monetary]])[0])
        clv_usd = max(0.0, clv_usd)
        model_used = "Machine Learning"
    elif ensure_deep_model_loaded():
        X_input = deep_clv_scaler.transform([[recency, frequency, monetary]])
        clv_usd = float(deep_clv_model.predict(X_input)[0])
        model_used = "Deep Learning (Auto)"
    elif ml_clv_available:
        clv_usd = float(ml_clv_model.predict([[recency, frequency, monetary]])[0])
        clv_usd = max(0.0, clv_usd)
        model_used = "Machine Learning (Auto)"
    else:
        flash("No CLV model artifacts found. Please train at least one model.")
        return render_template(
            "index.html",
            model_selected=model_selected,
            deep_clv_available=deep_clv_available,
            ml_clv_available=ml_clv_available,
            prediction_count=prediction_count
        )

    clv_inr = clv_usd * USD_TO_INR
    return render_template(
        "index.html",
        result=segment,
        r=recency,
        f=frequency,
        m=monetary,
        strategy=strategy,
        offer_badge=offer_badge,
        preferred_fabric=preferred_fabric,
        recency_score=rfm_scores["recency"],
        rfm_scores=rfm_scores,
        offer=offer,
        offer_qr_data=offer_qr_data,
        segment_label=segment_ui["label"],
        segment_theme=segment_ui["theme"],
        segment_icon=segment_ui["icon"],
        clv_usd=clv_usd,
        clv_inr=clv_inr,
        model_used=model_used,
        model_selected=model_selected,
        deep_clv_available=deep_clv_available,
        ml_clv_available=ml_clv_available,
        prediction_count=prediction_count
    )


@app.route("/fetch_customer", methods=["GET"])
def fetch_customer():
    customer_id = request.args.get("customer_id", "").strip()
    if not customer_id:
        return jsonify({"success": False, "message": "Customer ID is required."}), 400

    normalized_id = normalize_customer_id(customer_id)
    if normalized_id is None:
        return jsonify({"success": False, "message": "Invalid Customer ID format."}), 400

    if ensure_customer_lookup_loaded() is None:
        return jsonify({"success": False, "message": "Customer dataset not available on server."}), 500

    try:
        record = customer_lookup_by_id.get(normalized_id) if customer_lookup_by_id else None
        if record is None:
            return jsonify({"success": False, "message": "Customer ID not found in dataset."}), 404

        last_purchase_date = pd.to_datetime(record["last_purchase"]).date().isoformat()
        total_purchases = int(record["total_purchases"])
        total_spending = float(record["total_spending"])
    except Exception as exc:
        print(f"Fetch customer failed for ID {customer_id}: {exc}")
        return jsonify({"success": False, "message": "Failed to fetch customer history."}), 500

    return jsonify(
        {
            "success": True,
            "customer_id": customer_id,
            "last_purchase": last_purchase_date,
            "total_purchases": total_purchases,
            "total_spending": round(total_spending, 2),
        }
    )


@app.route("/dashboard")
def dashboard():
    ready = warm_dashboard_cache(force_refresh=False)
    if not ready:
        flash("Customer dataset not available.")
        return redirect(url_for("home"))

    interactive_available = has_interactive_dashboard_assets()

    return render_template(
        "dashboard.html",
        total_customers=total_customers_cache,
        segment_percentages=segment_percentages_cache,
        interactive=interactive_available
    )


# CSV upload route
@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        file = request.files["file"]
        if not file:
            flash("No file uploaded.")
            return redirect(request.url)
        df = pd.read_csv(file)
        # Minimal cleaning for demo
        if 'CustomerID' not in df.columns:
            flash("CSV must contain CustomerID column.")
            return redirect(request.url)
        df = df.dropna(subset=['CustomerID'])
        if 'InvoiceDate' in df.columns:
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        if 'Quantity' in df.columns and 'UnitPrice' in df.columns:
            df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
        global rfm_cache, total_customers_cache, segment_percentages_cache
        rfm = compute_rfm(df)
        rfm['Segment'] = rfm.apply(
            lambda row: assign_segment(row['Recency'], row['Frequency'], row['Monetary']),
            axis=1
        )
        rfm_cache = rfm
        total_customers_cache = len(rfm)
        segment_counts = rfm['Segment'].value_counts()
        segment_percentages_cache = {seg: round(100*count/total_customers_cache,1) for seg, count in segment_counts.items()}
        generate_dashboard_plots(rfm)
        generate_interactive_dashboard(rfm)
        interactive_available = has_interactive_dashboard_assets()
        return render_template(
            "dashboard.html",
            total_customers=total_customers_cache,
            segment_percentages=segment_percentages_cache,
            interactive=interactive_available
        )
    return render_template("upload.html")


# Model evaluation metrics (print at startup)
def print_model_metrics():
    df = load_clean_dataset()
    if df is None:
        print("Customer dataset not found; cannot compute metrics.")
        return
    cluster_model = joblib.load("models/kmeans_model.pkl")
    cluster_scaler = joblib.load("models/scaler.pkl")
    rfm = compute_rfm(df)
    rfm_scaled = cluster_scaler.transform(rfm)
    clusters = cluster_model.predict(rfm_scaled)
    sil_score = silhouette_score(rfm_scaled, clusters)
    print("Silhouette Score:", sil_score)
    print("Inertia:", cluster_model.inertia_)
    print("Cluster Distribution:", pd.Series(clusters).value_counts().to_dict())

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
