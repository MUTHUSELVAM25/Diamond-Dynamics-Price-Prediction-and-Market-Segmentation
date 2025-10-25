# ============================================================
# DIAMOND DYNAMICS - STREAMLIT APP (FIXED VERSION)
# Price Prediction & Market Segmentation
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diamond Dynamics",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# LOAD ALL MODELS AND METADATA
# ============================================================

@st.cache_resource
def load_models():
    """Load all saved models and metadata"""
    try:
        with open('models/best_model.pkl', 'rb') as f:
            reg_model = pickle.load(f)
        
        with open('models/cluster_model.pkl', 'rb') as f:
            cluster_model = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        with open('models/mean_price_per_carat.pkl', 'rb') as f:
            mean_price_per_carat = pickle.load(f)
        
        with open('models/cluster_columns.pkl', 'rb') as f:
            cluster_columns = pickle.load(f)
        
        with open('models/regression_features.pkl', 'rb') as f:
            regression_features = pickle.load(f)
        
        with open('models/cluster_names.pkl', 'rb') as f:
            cluster_names = pickle.load(f)
        
        return (reg_model, cluster_model, scaler, mean_price_per_carat, 
                cluster_columns, regression_features, cluster_names)
    
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()

# Load models
(reg_model, cluster_model, scaler, mean_ppc, 
 cluster_cols, reg_features, CLUSTER_NAMES) = load_models()

st.success("✅ Models loaded successfully!")

# ============================================================
# MAIN UI
# ============================================================

st.title("💎 Diamond Dynamics: Price Prediction & Market Segmentation")
st.markdown("**Predict diamond prices and identify market segments using Machine Learning**")

# ============================================================
# SIDEBAR - USER INPUTS
# ============================================================

st.sidebar.header("📝 Enter Diamond Features")
st.sidebar.markdown("Fill in the diamond characteristics below:")

# Numeric Inputs
carat = st.sidebar.number_input("Carat", min_value=0.2, max_value=5.0, value=1.0, step=0.01, help="Weight of the diamond")
depth = st.sidebar.number_input("Depth %", min_value=50.0, max_value=70.0, value=61.0, step=0.1, help="Depth percentage")
table = st.sidebar.number_input("Table %", min_value=50.0, max_value=70.0, value=57.0, step=0.1, help="Table percentage")
x = st.sidebar.number_input("Length (x) mm", min_value=0.0, max_value=10.0, value=5.0, step=0.1, help="Length in mm")
y = st.sidebar.number_input("Width (y) mm", min_value=0.0, max_value=10.0, value=5.0, step=0.1, help="Width in mm")
z = st.sidebar.number_input("Height (z) mm", min_value=0.0, max_value=10.0, value=3.0, step=0.1, help="Height in mm")

# Categorical Inputs
cut = st.sidebar.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=4)
color = st.sidebar.selectbox("Color", ["J", "I", "H", "G", "F", "E", "D"], index=3)
clarity = st.sidebar.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=4)

# Encoding mappings (matching your OrdinalEncoder order)
cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

cut_encoded = cut_mapping[cut]
color_encoded = color_mapping[color]
clarity_encoded = clarity_mapping[clarity]

# ============================================================
# MODULE SELECTION
# ============================================================

st.markdown("---")
st.header("🔍 Choose Module:")

module = st.radio(
    "",
    ["💰 Price Prediction", "🍀 Market Segment Prediction"],
    horizontal=True
)

# ============================================================
# MODULE 1: PRICE PREDICTION
# ============================================================

if module == "💰 Price Prediction":
    st.subheader("💰 Predict Diamond Price (INR)")
    st.markdown("Get an estimated price for your diamond based on its features.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Predict Price", type="primary", use_container_width=True):
            try:
                # Step 1: Log transformation (as done during training)
                carat_log = np.log1p(carat)
                x_log = np.log1p(x)
                y_log = np.log1p(y)
                z_log = np.log1p(z)
                
                # Step 2: Create derived features
                Volume = x_log * y_log * z_log
                PricePerCarat = mean_ppc  # Use training mean
                DimensionRatio = (x_log + y_log) / (2 * z_log) if z_log != 0 else 0
                
                # Step 3: Build feature dictionary
                feature_dict = {
                    'carat': carat_log,
                    'x': x_log,
                    'y': y_log,
                    'z': z_log,
                    'depth': depth,
                    'table': table,
                    'cut': cut_encoded,
                    'color': color_encoded,
                    'clarity': clarity_encoded,
                    'Volume': Volume,
                    'PricePerCarat': PricePerCarat,
                    'DimensionRatio': DimensionRatio
                }
                
                # Step 4: Create input DataFrame with only the features your model needs
                input_data = pd.DataFrame({
                    feature: [feature_dict[feature]] 
                    for feature in reg_features
                })
                
                # Debug: Show what features we're using
                with st.expander("🐛 Debug Information"):
                    st.write(f"**Expected features:** {reg_features}")
                    st.write(f"**Input data columns:** {input_data.columns.tolist()}")
                    st.write(f"**Input data shape:** {input_data.shape}")
                    st.write(f"**Input values:**")
                    st.dataframe(input_data)
                
                # Step 5: Predict
                predicted_price_log = reg_model.predict(input_data)[0]
                
                # Step 6: Inverse log transformation
                predicted_price_usd = np.expm1(predicted_price_log)
                
                # Step 7: Convert to INR
                usd_to_inr = 83.0
                predicted_price_inr = predicted_price_usd * usd_to_inr
                
                # Display results
                st.success(f"✅ **Predicted Diamond Price: ₹{predicted_price_inr:,.2f} INR**")
                st.info(f"💵 Equivalent to ${predicted_price_usd:,.2f} USD")
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    st.write(f"**Carat:** {carat} | **Cut:** {cut} | **Color:** {color} | **Clarity:** {clarity}")
                    st.write(f"**Dimensions:** {x} × {y} × {z} mm")
                    st.write(f"**Depth:** {depth}% | **Table:** {table}%")
                
            except Exception as e:
                st.error(f"❌ Prediction Error: {e}")
                with st.expander("🐛 Debug Information"):
                    st.write(f"**Error:** {str(e)}")
                    st.write(f"**Expected features:** {reg_features}")
                    st.write(f"**Feature dict keys:** {list(feature_dict.keys())}")
    
    with col2:
        st.info("💡 **Tip:** Adjust the diamond features in the sidebar to see different price estimates.")

# ============================================================
# MODULE 2: MARKET SEGMENT PREDICTION
# ============================================================

elif module == "🍀 Market Segment Prediction":
    st.subheader("🍀 Predict Market Segment (Clustering)")
    st.markdown("Identify which market segment your diamond belongs to.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔮 Predict Segment", type="primary", use_container_width=True):
            try:
                # Step 1: Log transformation
                carat_log = np.log1p(carat)
                x_log = np.log1p(x)
                y_log = np.log1p(y)
                z_log = np.log1p(z)
                
                # Step 2: Create derived features
                Volume = x_log * y_log * z_log
                DimensionRatio = (x_log + y_log) / (2 * z_log) if z_log != 0 else 0
                PricePerCarat = mean_ppc
                
                # Step 3: Build feature dictionary
                feature_dict = {
                    'carat': carat_log,
                    'x': x_log,
                    'y': y_log,
                    'z': z_log,
                    'depth': depth,
                    'table': table,
                    'cut': cut_encoded,
                    'color': color_encoded,
                    'clarity': clarity_encoded,
                    'Volume': Volume,
                    'PricePerCarat': PricePerCarat,
                    'DimensionRatio': DimensionRatio
                }
                
                # Step 4: Create input DataFrame with ALL cluster features
                cluster_input = pd.DataFrame({
                    col: [feature_dict[col]] 
                    for col in cluster_cols
                })
                
                # Debug: Show what features we're using
                with st.expander("🐛 Debug Information"):
                    st.write(f"**Expected features:** {cluster_cols}")
                    st.write(f"**Input data columns:** {cluster_input.columns.tolist()}")
                    st.write(f"**Input data shape:** {cluster_input.shape}")
                    st.write(f"**Input values:**")
                    st.dataframe(cluster_input)
                
                # Step 5: Scale the input
                cluster_input_scaled = scaler.transform(cluster_input)
                
                # Step 6: Predict cluster
                cluster_label = cluster_model.predict(cluster_input_scaled)[0]
                cluster_name = CLUSTER_NAMES.get(cluster_label, f"Cluster {cluster_label}")
                
                # Display results
                st.success(f"✅ **Market Segment: {cluster_name}**")
                st.info(f"🏷️ Cluster Number: {cluster_label}")
                
                # Show segment description
                segment_descriptions = {
                    "Premium Heavy Diamonds": "🌟 High-value diamonds with larger carat weight and superior quality.",
                    "Affordable Small Diamonds": "💎 Budget-friendly diamonds perfect for everyday jewelry.",
                    "Mid-range Balanced Diamonds": "⚖️ Well-balanced diamonds offering good value for money."
                }
                
                if cluster_name in segment_descriptions:
                    st.markdown(f"**Description:** {segment_descriptions[cluster_name]}")
                
                # Show input summary
                with st.expander("📋 Input Summary"):
                    st.write(f"**Carat:** {carat} | **Cut:** {cut} | **Color:** {color} | **Clarity:** {clarity}")
                    st.write(f"**Dimensions:** {x} × {y} × {z} mm")
                    st.write(f"**Depth:** {depth}% | **Table:** {table}%")
                
            except Exception as e:
                st.error(f"❌ Clustering Error: {e}")
                with st.expander("🐛 Debug Information"):
                    st.write(f"**Error:** {str(e)}")
                    st.write(f"**Expected features:** {cluster_cols}")
                    st.write(f"**Feature dict keys:** {list(feature_dict.keys())}")
    
    with col2:
        st.info("💡 **Tip:** Market segments help you understand your diamond's positioning in the market.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>💎 <b>Diamond Dynamics</b> - ML-Powered Diamond Analysis</p>
    <p>Built with Streamlit • Powered by Scikit-learn & XGBoost</p>
</div>
""", unsafe_allow_html=True)
