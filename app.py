import streamlit as st
import pandas as pd
import joblib
import os
import io
import base64
import importlib
from sklearn.metrics import silhouette_score

# If you still have your src folder, these will work.
# We will just display the images/html they generate.
try:
    from src.visualization import generate_dashboard_plots
    from src.interactive_dashboard import generate_interactive_dashboard
except ImportError:
    pass

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="Textile Customer Segmentation", layout="wide", page_icon="🛍️")

# --- Constants & Dictionaries ---
USD_TO_INR = 83

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

# --- Caching Data & Models (The Streamlit Way) ---
@st.cache_data
def load_clean_dataset():
    try:
        df = pd.read_excel("dataset/Online Retail.xlsx")
        df = df.dropna(subset=["CustomerID"])
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_thresholds():
    try:
        return joblib.load("models/rfm_thresholds.pkl")
    except FileNotFoundError:
        df = load_clean_dataset()
        if df is None:
            # Fallback default thresholds if no data or model exists
            return {"recency": 90.0, "frequency": 5.0, "monetary": 1000.0}
        
        snapshot_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = df.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
            'InvoiceNo': 'nunique',
            'TotalPrice': 'sum'
        })
        rfm.columns = ['Recency','Frequency','Monetary']
        
        thresholds = {
            "recency": float(rfm['Recency'].quantile(0.75)),
            "frequency": float(rfm['Frequency'].quantile(0.50)),
            "monetary": float(rfm['Monetary'].quantile(0.50)),
        }
        os.makedirs("models", exist_ok=True)
        joblib.dump(thresholds, "models/rfm_thresholds.pkl")
        return thresholds

@st.cache_data
def get_customer_lookup():
    df = load_clean_dataset()
    if df is None:
        return None
    summary = df.groupby("CustomerID").agg(
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
        total_purchases=("InvoiceNo", "nunique"),
        total_spending=("TotalPrice", "sum"),
    )
    summary["first_purchase"] = summary["first_purchase"].dt.normalize()
    summary["last_purchase"] = summary["last_purchase"].dt.normalize()
    return summary

# Load globals
thresholds = load_thresholds()
customer_lookup = get_customer_lookup()

# --- Helper Functions ---
def assign_segment(recency, frequency, monetary):
    if (recency <= thresholds["recency"] and 
        frequency > thresholds["frequency"] and 
        monetary > thresholds["monetary"]):
        return "High Value Customer"
    if recency > thresholds["recency"]:
        return "Lost Customer"
    if frequency > thresholds["frequency"]:
        return "Regular Customer"
    return "Occasional Customer"

def calculate_clv(total_spending, total_purchases, first_purchase_date=None, last_purchase_date=None, future_lifespan_years=3.0):
    if total_purchases == 0:
        return 0.0
    avg_purchase_value = total_spending / total_purchases
    lifespan_years = 3.0
    
    if first_purchase_date is not None and last_purchase_date is not None:
        try:
            first_date = pd.to_datetime(first_purchase_date)
            last_date = pd.to_datetime(last_purchase_date)
            lifespan_days = (last_date - first_date).days
            if lifespan_days > 0:
                lifespan_years = lifespan_days / 365.25
        except Exception:
            pass

    if lifespan_years <= 0:
        lifespan_years = 1.0

    purchase_frequency = total_purchases / lifespan_years
    clv = avg_purchase_value * purchase_frequency * future_lifespan_years
    clv *= 1.10  # 10% growth multiplier
    return round(clv, 2)

def build_offer_payload(segment, customer_id):
    customer_part = str(customer_id).replace(" ", "").replace("-", "")[-4:] or "0000"
    segment_code = "".join([s[0] for s in segment.split()][:3]).upper()
    coupon_code = f"{segment_code}{customer_part}26"
    return coupon_code

def generate_qr_code(coupon_code):
    qr_payload = f"TEXTILE-OFFER:{coupon_code}"
    try:
        qrcode_module = importlib.import_module("qrcode")
        qr = qrcode_module.QRCode(version=1, box_size=4, border=1)
        qr.add_data(qr_payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except ModuleNotFoundError:
        return None

# --- Session State for Login ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# --- App Routing & UI ---
if not st.session_state["logged_in"]:
    st.title("🔒 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if username == "admin" and password == "admin":
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("Invalid credentials. Use admin / admin.")
else:
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Predict Offer", "Dashboard", "Customer Evolution", "Upload Data"])
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

    # --- Predict Module ---
    if page == "Predict Offer":
        st.title("🎯 Customer Prediction & Offers")
        
        # Simple Fetch Mechanism
        col1, col2 = st.columns([3, 1])
        with col1:
            fetch_id = st.text_input("Fetch Existing Customer Data (Enter ID):")
        with col2:
            st.write("") # Spacing
            st.write("")
            fetch_clicked = st.button("Fetch")

        # Defaults for form
        def_id = ""
        def_date = pd.Timestamp.today().date()
        def_purchases = 0.0
        def_spending = 0.0
        def_first_purchase = None

        if fetch_clicked and fetch_id and customer_lookup is not None:
            try:
                numeric_id = float(fetch_id)
                if numeric_id in customer_lookup.index:
                    record = customer_lookup.loc[numeric_id]
                    def_id = str(fetch_id)
                    def_date = record["last_purchase"].date()
                    def_purchases = float(record["total_purchases"])
                    def_spending = float(record["total_spending"])
                    def_first_purchase = record["first_purchase"]
                    st.success(f"Loaded data for Customer {fetch_id}")
                else:
                    st.warning("Customer ID not found in dataset.")
            except ValueError:
                st.error("Invalid ID format.")

        with st.form("prediction_form"):
            customer_id = st.text_input("Customer ID", value=def_id)
            last_purchase = st.date_input("Last Purchase Date", value=def_date)
            total_purchases = st.number_input("Total Purchases", min_value=0.0, value=def_purchases)
            total_spending = st.number_input("Total Spending", min_value=0.0, value=def_spending)
            
            submit = st.form_submit_button("Analyze Customer")

        if submit:
            # RFM Calculations
            reference_date = pd.Timestamp.today().normalize()
            if customer_lookup is not None:
                reference_date = customer_lookup["last_purchase"].max() + pd.Timedelta(days=1)
                
            recency = max((reference_date - pd.to_datetime(last_purchase)).days, 0)
            segment = assign_segment(recency, total_purchases, total_spending)
            
            clv_usd = calculate_clv(total_spending, total_purchases, def_first_purchase, pd.to_datetime(last_purchase))
            clv_inr = clv_usd * USD_TO_INR

            st.markdown("---")
            st.subheader(f"Segment: {segment}")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Recency (Days)", f"{recency:.0f}")
            c2.metric("Frequency", f"{total_purchases:.0f}")
            c3.metric("Monetary", f"₹ {total_spending:,.2f}")
            c4.metric("Est. CLV", f"₹ {clv_inr:,.2f}")

            st.info(f"**Strategy:** {strategies.get(segment, 'N/A')}")
            st.success(f"**Offer:** {segment_offer_badges.get(segment, 'Standard Offer')}")
            st.text(f"**Preferred Fabric:** {segment_fabric_preferences.get(segment, 'Mixed')}")
            
            # QR Code Generation
            coupon_code = build_offer_payload(segment, customer_id)
            st.write(f"**Coupon Code:** `{coupon_code}`")
            
            qr_bytes = generate_qr_code(coupon_code)
            if qr_bytes:
                st.image(qr_bytes, caption="Scan Offer at Counter", width=150)


    # --- Dashboard Module ---
    elif page == "Dashboard":
        st.title("📊 Analytics Dashboard")
        
        df = load_clean_dataset()
        if df is not None:
            st.write(f"**Total Valid Transactions:** {len(df):,}")
            
            # If your custom src scripts successfully rendered plots to /static, we can show them
            st.subheader("Visualizations")
            col1, col2 = st.columns(2)
            
            if os.path.exists("static/segment_pie.png"):
                col1.image("static/segment_pie.png", caption="Segment Distribution")
            else:
                col1.info("Pie chart not generated yet. Ensure `src.visualization` is working.")
                
            if os.path.exists("static/cluster_plot.png"):
                col2.image("static/cluster_plot.png", caption="Customer Clusters")

            if os.path.exists("static/segment_pie_interactive.html"):
                st.components.v1.html(open("static/segment_pie_interactive.html", 'r').read(), height=400)
        else:
            st.warning("Customer dataset not available on the server. Please upload data.")


    # --- Evolution Module ---
    elif page == "Customer Evolution":
        st.title("📈 Customer Evolution Timeline")
        
        cust_id = st.text_input("Enter Customer ID for Timeline:")
        if st.button("Generate Timeline") and cust_id:
            df = load_clean_dataset()
            if df is None:
                st.error("Dataset missing")
            else:
                try:
                    numeric_id = float(cust_id)
                    cust_df = df[df["CustomerID"] == numeric_id].copy()
                    
                    if cust_df.empty:
                        st.warning("Customer not found in dataset.")
                    else:
                        cust_df = cust_df.sort_values("InvoiceDate")
                        cust_df["Month"] = cust_df["InvoiceDate"].dt.to_period('M')
                        
                        months = sorted(cust_df["Month"].unique())
                        cum_spending = 0
                        cum_purchases = 0
                        first_purchase = cust_df["InvoiceDate"].min()
                        
                        timeline_data = []
                        
                        for m in months:
                            m_df = cust_df[cust_df["Month"] == m]
                            cum_spending += (m_df["Quantity"] * m_df["UnitPrice"]).sum()
                            cum_purchases += m_df["InvoiceNo"].nunique()
                            last_p = m_df["InvoiceDate"].max()
                            
                            clv = calculate_clv(cum_spending, cum_purchases, first_purchase, last_p)
                            clv_inr = clv * USD_TO_INR * 5 # Applying your boost factor
                            
                            if clv_inr < 100000:
                                seg = "Low"
                            elif clv_inr <= 300000:
                                seg = "Medium"
                            else:
                                seg = "High"
                                
                            timeline_data.append({
                                "Month": m.strftime('%b %Y'),
                                "Est. CLV (INR)": round(clv_inr, 2),
                                "Segment": seg
                            })
                            
                        timeline_df = pd.DataFrame(timeline_data)
                        st.dataframe(timeline_df, use_container_width=True)
                        st.line_chart(timeline_df.set_index("Month")["Est. CLV (INR)"])
                        
                except ValueError:
                    st.error("Invalid Customer ID")


    # --- Upload Module ---
    elif page == "Upload Data":
        st.title("📁 Upload New Dataset")
        
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            st.info("File uploaded successfully. In a production environment, you would add logic here to save this over `dataset/Online Retail.xlsx` and clear the Streamlit caches using `st.cache_data.clear()`.")
            # Example of how you would load it to preview:
            if uploaded_file.name.endswith('.csv'):
                preview = pd.read_csv(uploaded_file).head()
            else:
                preview = pd.read_excel(uploaded_file).head()
            st.dataframe(preview)
