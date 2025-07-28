import streamlit as st
import pickle
import pandas as pd

# Load models and data
kmeans = pickle.load(open("models/rfm_kmeans_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
sim_matrix = pickle.load(open("models/product_similarity.pkl", "rb"))
df = pd.read_csv("data/online_retail.csv", encoding='ISO-8859-1')

st.set_page_config(page_title="🛍️ Shopper Spectrum", layout="wide")

# CSS Styling for sidebar and active buttons
st.markdown("""
    <style>
        /* Sidebar background and layout */
        .sidebar-box {
            background-color: #fff;
            border-radius: 12px;
            padding: 20px;
            margin-top: 30px;
            width: 100%;
        }
        .nav-btn {
            display: block;
            width: 100%;
            text-align: center;
            padding: 10px 16px;
            margin: 10px 0;
            font-weight: bold;
            border-radius: 8px;
            color: #000;
            background-color: #f0f0f0;
            border: none;
            cursor: pointer;
            transition: 0.2s ease-in-out;
        }
        .nav-btn:hover {
            background-color: #fa5252;
            color: #fff;
        }
        .active-nav {
            background-color: #fa5252 !important;
            color: #fff !important;
        }
        .center-content {
            display: flex;
            justify-content: center;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize nav state
if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"

# Sidebar with centered buttons
with st.sidebar:
    st.markdown("<div class='sidebar-box'>", unsafe_allow_html=True)
    st.markdown("<div class='center-content'>", unsafe_allow_html=True)

    def nav_button(label, emoji):
        selected = st.session_state["nav"] == label
        button_html = f"""
            <button class="nav-btn {'active-nav' if selected else ''}" onclick="window.location.reload();">{emoji} {label}</button>
            <script>
                const navButton = window.parent.document.querySelectorAll('.nav-btn');
                navButton.forEach((btn, i) => {{
                    btn.addEventListener('click', () => {{
                        window.parent.localStorage.setItem('navChoice', '{label}');
                    }});
                }});
            </script>
        """
        st.markdown(button_html, unsafe_allow_html=True)

    nav_button("Home", "🏠")
    nav_button("Clustering", "📊")
    nav_button("Recommendation", "📦")

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Manual nav update from JS simulation
nav_choice = st.query_params.get("navChoice", [st.session_state["nav"]])[0]
st.session_state["nav"] = nav_choice


# --- Title ---
st.markdown("<h1 style='text-align: center;'>🛍️ Shopper Spectrum</h1>", unsafe_allow_html=True)

# --- Home Page ---
if st.session_state["nav"] == "Home":
    st.subheader("📌 Welcome to Shopper Spectrum")
    st.markdown("""
    This interactive tool allows you to:

    - 🎯 **Segment Customers** using Recency, Frequency, and Monetary (RFM) values.
    - 🤖 **Predict customer segments** like High-Value, Occasional, At-Risk, etc.
    - 🧠 **Recommend products** similar to a chosen one using product name input.

    <br>
    Use the sidebar to navigate between:
    - 📊 Clustering
    - 📦 Recommendations
    """, unsafe_allow_html=True)

# --- Clustering UI ---
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

        st.markdown(f"<h4 style='color: green;'>2</h4>", unsafe_allow_html=True)
        st.markdown(f"<p>This customer belongs to: <b>{segment_map.get(label, 'Unknown')}</b></p>", unsafe_allow_html=True)

# --- Recommendation UI ---
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
                st.warning("Similarity data not found for this product.")
        else:
            st.warning("Product name not found in database.")
