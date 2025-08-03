import streamlit as st
import pickle
import pandas as pd

# Load models and data
kmeans = pickle.load(open("models/rfm_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("models/rfm_scaler.pkl", "rb"))
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
    st.markdown("<h3 style='text-align: center;'>Navigate</h3>", unsafe_allow_html=True)
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
        0: "Regular Buyer",
        1: "At-Risk Customer",
        2: "High-Value Customer",
        3: "Occasional Shopper"
    }


        st.success(f"This customer belongs to: **{segment_map.get(label, 'Unknown')}**")

# --- Recommendation Page ---
# --- Recommendation Page ---
elif st.session_state["nav"] == "Recommendation":
    st.subheader("🔍 Get Product Recommendations")

    # Ensure sim_matrix uses string index/columns
    sim_matrix.columns = sim_matrix.columns.astype(str)
    sim_matrix.index = sim_matrix.index.astype(str)

    # Normalize product names for matching
    df["Description"] = df["Description"].astype(str).str.strip()

    product_name_input = st.text_input("Enter Product Name (Description):")

    if st.button("Get Recommendations") and product_name_input.strip():
        product_name_input = product_name_input.strip()

        # Try exact match
        matching_products = df[df['Description'].str.lower() == product_name_input.lower()]

        # Optional fuzzy fallback
        if matching_products.empty:
            from fuzzywuzzy import process
            all_descriptions = df['Description'].dropna().unique()
            best_match, score = process.extractOne(product_name_input, all_descriptions)
            if score > 80:
                st.info(f"No exact match found. Using closest match: **{best_match}** (score: {score})")
                product_name_input = best_match
            else:
                st.warning("❌ No suitable match found in product descriptions.")
                st.stop()

        st.write("Matched Product:", product_name_input)

        if product_name_input in sim_matrix.columns:
            top_similar = sim_matrix[product_name_input].sort_values(ascending=False)[1:6].index.tolist()

            st.markdown("### 🧾 Recommended Products:")
            for i, desc in enumerate(top_similar):
                st.markdown(f"{i+1}. {desc}")
        else:
            st.warning("⚠️ Similarity data not found for this product.")
