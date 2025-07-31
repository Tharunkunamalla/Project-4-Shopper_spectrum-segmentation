import streamlit as st
import pickle
import pandas as pd

# Load models and data
kmeans = pickle.load(open("models/rfm_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
sim_matrix = pickle.load(open("models/product_similarity.pkl", "rb"))
df = pd.read_csv("data/online_retail.csv", encoding='ISO-8859-1')

# Set Streamlit page config
st.set_page_config(page_title="🛍️ Shopper Spectrum", layout="wide")

# CSS Styling
st.markdown("""
    <style>
        .sidebar-box {
            background-color: #fff;
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
        }
        .center-content {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Session state for navigation
if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>🧭 Navigate</h3>", unsafe_allow_html=True)
    page = st.radio(
        "Go to",
        ["Home", "Clustering", "Recommendation"],
        index=["Home", "Clustering", "Recommendation"].index(st.session_state["nav"])
    )
    st.session_state["nav"] = page
    st.markdown("</div>", unsafe_allow_html=True)

# --- App Title ---
st.markdown("<h1 style='text-align: center;'>🛍️ Shopper Spectrum</h1>", unsafe_allow_html=True)

# --- Home Page ---
if st.session_state["nav"] == "Home":
    st.subheader("📌 Welcome to Shopper Spectrum")
    st.markdown("""
    This interactive tool allows you to:

    - 🎯 **Segment Customers** using Recency, Frequency, and Monetary (RFM) values.
    - 🤖 **Predict customer segments** like High-Value, Occasional, At-Risk, etc.
    - 🧠 **Recommend products** similar to a chosen one using product name input.

    Use the sidebar to navigate between:
    - 📊 Clustering
    - 📦 Recommendations
    """, unsafe_allow_html=True)

# --- Clustering Page ---
elif st.session_state["nav"] == "Clustering":
    st.subheader("👥 Customer Segmentation")

    recency = st.number_input("Recency (days since last purchase)", min_value=0)
    frequency = st.number_input("Frequency (number of purchases)", min_value=0)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0)

    if st.button("Predict Segment"):
        input_scaled = scaler.transform([[recency, frequency, monetary]])
        label = kmeans.predict(input_scaled)[0]

        segment_map = {
            0: "At-Risk Customer",
            1: "Regular Buyer",
            2: "High-Value Customer",
            3: "Occasional Shopper"
        }

        st.success(f"This customer belongs to: **{segment_map.get(label, 'Unknown')}**")

# --- Recommendation Page ---
elif st.session_state["nav"] == "Recommendation":
    st.subheader("🔍 Get Product Recommendations")

    product_name_input = st.text_input("Enter Product Name (Description):")

    if st.button("Get Recommendations"):
        matching_products = df[df['Description'].str.lower() == product_name_input.lower()]

        if not matching_products.empty:
            product_code = matching_products["StockCode"].values[0]

            if product_code in sim_matrix.columns:
                top_similar = sim_matrix[product_code].sort_values(ascending=False)[1:6].index.tolist()

                st.markdown("### Recommended Product Names:")
                for i, code in enumerate(top_similar):
                    similar_desc = df[df["StockCode"] == code]["Description"].dropna().unique()
                    if similar_desc.any():
                        st.markdown(f"{i+1}. {similar_desc[0]}")
            else:
                st.warning("⚠️ Similarity data not found for this product.")
        else:
            st.warning("⚠️ Product name not found in database.")
