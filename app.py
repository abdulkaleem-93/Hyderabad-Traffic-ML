import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go  # ✅ FIXED - Add this import

# Config
st.set_page_config(page_title="🚦 Hyderabad Traffic Predictor", layout="wide", page_icon="🚦")

st.title("🚦 Hyderabad Traffic Prediction System")
st.markdown("**XGBoost Production Model | RMSE: 0.83 | 50K Hyderabad Data**")

# Load models
@st.cache_resource
def load_models():
    model = joblib.load('traffic_xgb_model.pkl')
    scaler = joblib.load('traffic_scaler.pkl')
    le = joblib.load('location_encoder.pkl')
    return model, scaler, le

model, scaler, le = load_models()

# === 1. LIVE PREDICTION ===
st.header("🔮 **Live Prediction**")
col1, col2, col3, col4 = st.columns(4)
with col1: hour = st.slider("Hour", 0, 23, 18)
with col2: location = st.selectbox("Location", le.classes_)
with col3: rain = st.checkbox("🌧️ Heavy Rain") 
with col4: accident = st.checkbox("🚑 Accident")

if st.button("🚦 **PREDICT TRAFFIC**", type="primary"):
    input_data = pd.DataFrame({
        'hour': [hour], 'day_of_week': [1], 'month': [6],
        'is_rush_hour': [1 if hour in [17,18,19] else 0],
        'is_weekend': [0], 'heavy_rain': [int(rain)],
        'hot_weather': [0], 'speed_ratio': [2000],
        'location_encoded': [le.transform([location])[0]],
        'rush_rain_interaction': [int(rain and hour in [17,18,19])],
        'volume_per_speed': [2000], 'metro_nearby': [0],
        'accidents_count': [int(accident)], 'volume_3h_avg': [2500]
    })
    
    X_scaled = scaler.transform(input_data)
    prediction = model.predict(X_scaled)[0]
    
    status = "🟢 CLEAR" if prediction < 1.5 else "🟡 MODERATE" if prediction < 2.5 else "🔴 JAM"
    st.metric("🚦 Congestion Score", f"{prediction:.2f}", f"{status}")
    st.success(f"**{location} at {hour}:00** → **{status}** ({prediction:.2f})")

# === 2. FULLY INTERACTIVE MAP ===
st.markdown("---")
st.header("🗺️ **Interactive Hyderabad Traffic Map**")

traffic_data = {
    'location': ['HITEC City', 'Gachibowli', 'Raidurg', 'Banjara Hills', 'Jubilee Hills', 'Madhapur', 'Cyber Towers'],
    'lat': [17.4400, 17.4123, 17.4286, 17.4200, 17.4286, 17.4320, 17.4410],
    'lon': [78.3800, 78.3501, 78.3987, 78.4417, 78.4100, 78.3890, 78.3870],
    'score': [2.15, 1.92, 1.78, 1.65, 1.88, 1.95, 2.05],
    'status': ['🔴 JAM', '🟡 MODERATE', '🟡 MODERATE', '🟢 CLEAR', '🟡 MODERATE', '🟡 MODERATE', '🟡 MODERATE']
}
df_map = pd.DataFrame(traffic_data)

# ✅ PERFECT INTERACTIVE MAP
fig = px.scatter_mapbox(
    df_map,
    lat='lat',
    lon='lon',
    size='score',
    color='score',
    hover_name='location',
    hover_data=['status'],
    color_continuous_scale='RdYlGn_r',
    size_max=25,
    zoom=11,
    height=600,
    mapbox_style="open-street-map"
)

# FIXED: Use dict instead of go.layout.mapbox.Center
fig.update_layout(
    mapbox=dict(
        center=dict(lat=17.3850, lon=78.4867),  # ✅ FIXED - No 'go' needed
        style="open-street-map",
        zoom=11
    ),
    height=600,
    margin={"r":0,"t":40,"l":0,"b":0},
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True, config={
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': ['pan2d','zoom2d','zoomIn2d','zoomOut2d','autoScale2d','resetScale2d']
})

# Map instructions
with st.expander("🖱️ **Map Controls**"):
    st.markdown("""
    ✅ **Hover** → Location details  
    ✅ **Scroll** → Zoom in/out
    ✅ **Drag** → Pan map
    ✅ **Toolbar** → Top-right controls
    """)

# === 3. METRICS ===
col1, col2, col3, col4 = st.columns(4)
col1.metric("✅ RMSE", "0.83")
col2.metric("📊 Dataset", "50K")
col3.metric("⚙️ Features", "14")
col4.metric("🎯 Clusters", "3")

st.markdown("---")
st.markdown("""
# 🎓 **Production ML Portfolio**
✅ End-to-end ML pipeline | ✅ XGBoost RMSE 0.83  
✅ Live predictions | ✅ Interactive Hyderabad map  
✅ Production deployment ready
""")
st.caption("**Hyderabad ML Engineer | XGBoost + Streamlit + Plotly | Feb 2026** 🚀")
