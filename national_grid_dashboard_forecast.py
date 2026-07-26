import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import time
import plotly.graph_objects as go
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from datetime import datetime, timedelta

# =========================================================
# 1. PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="🇰🇪 National Grid Twin (Forecast API)",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("⚡ Kenya National Grid Twin (Weather Forecast Edition)")
st.markdown("### Real-Time Demand Forecasting using LIVE Weather Forecasts")

# =========================================================
# 2. LOAD MODEL & SCALER
# =========================================================
@st.cache_resource(show_spinner=False)
def load_assets():
    model = load_model(
        'national_grid_twin.h5',
        custom_objects={'mse': MeanSquaredError()}
    )
    with open('national_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_assets()

# =========================================================
# 3. REGION CONFIGURATION
# =========================================================
regions = {
    'Nairobi': {'city': 'Nairobi', 'lat': -1.286389, 'lon': 36.817223},
    'Coast': {'city': 'Mombasa', 'lat': -4.0435, 'lon': 39.6682},
    'North_Eastern': {'city': 'Garissa', 'lat': -0.4536, 'lon': 39.6462},
    'Central_Rift': {'city': 'Nakuru', 'lat': -0.3031, 'lon': 36.0800},
    'North_Rift': {'city': 'Eldoret', 'lat': 0.5143, 'lon': 35.2698},
    'Mt_Kenya': {'city': 'Nyeri', 'lat': -0.4167, 'lon': 36.9500},
    'West_Kenya': {'city': 'Kisumu', 'lat': -0.0917, 'lon': 34.7680},
    'South_Nyanza': {'city': 'Kisii', 'lat': -0.6779, 'lon': 34.7793}
}

region_order = list(regions.keys())

# =========================================================
# 4. EXACT FEATURE LISTS
# =========================================================
weather_features = [
    'Nairobi_temp_C', 'Nairobi_solar_W_m2',
    'Mombasa_temp_C', 'Mombasa_solar_W_m2',
    'Garissa_temp_C', 'Garissa_solar_W_m2',
    'Nakuru_temp_C', 'Nakuru_solar_W_m2',
    'Eldoret_temp_C', 'Eldoret_solar_W_m2',
    'Nyeri_temp_C', 'Nyeri_solar_W_m2',
    'Kisumu_temp_C', 'Kisumu_solar_W_m2',
    'Kisii_temp_C', 'Kisii_solar_W_m2'
]

time_features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']
ALL_FEATURES = weather_features + time_features

DEMAND_COLUMNS = [
    'Nairobi_demand_MW', 'Coast_demand_MW', 'North_Eastern_demand_MW',
    'Central_Rift_demand_MW', 'North_Rift_demand_MW', 'Mt_Kenya_demand_MW',
    'West_Kenya_demand_MW', 'South_Nyanza_demand_MW'
]

# =========================================================
# 5. FETCH HISTORICAL & FORECAST DATA
# =========================================================
def fetch_weather_data():
    now = datetime.now()
    
    # 1. Fetch 72 hours of PAST data (for context)
    past_end = now.strftime("%Y-%m-%d")
    past_start = (now - timedelta(hours=72)).strftime("%Y-%m-%d")
    
    # 2. Fetch 24 hours of FUTURE data (Forecast)
    future_end = (now + timedelta(hours=24)).strftime("%Y-%m-%d")
    future_start = now.strftime("%Y-%m-%d")
    
    all_past = {}
    all_future = {}
    
    for region, info in regions.items():
        lat, lon = info['lat'], info['lon']
        
        # --- Past Data (Archive API) ---
        url_past = (
            f"https://archive-api.open-meteo.com/v1/archive?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={past_start}&end_date={past_end}"
            f"&hourly=temperature_2m,shortwave_radiation"
        )
        try:
            resp = requests.get(url_past, timeout=10)
            if resp.status_code == 200:
                data = resp.json()['hourly']
                df = pd.DataFrame({
                    'datetime': data['time'],
                    f"{info['city']}_temp_C": data['temperature_2m'],
                    f"{info['city']}_solar_W_m2": data['shortwave_radiation']
                })
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                all_past[region] = df
            else:
                st.error(f"❌ Past API Error for {region}")
                return None, None
        except Exception as e:
            st.error(f"❌ Past Connection error {region}: {e}")
            return None, None

        # --- Future Data (Forecast API) ---
        url_future = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}"
            f"&start_date={future_start}&end_date={future_end}"
            f"&hourly=temperature_2m,shortwave_radiation"
            f"&timezone=Africa/Nairobi"
        )
        try:
            resp = requests.get(url_future, timeout=10)
            if resp.status_code == 200:
                data = resp.json()['hourly']
                df = pd.DataFrame({
                    'datetime': data['time'],
                    f"{info['city']}_temp_C": data['temperature_2m'],
                    f"{info['city']}_solar_W_m2": data['shortwave_radiation']
                })
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                all_future[region] = df
            else:
                st.error(f"❌ Forecast API Error for {region}")
                return None, None
        except Exception as e:
            st.error(f"❌ Forecast Connection error {region}: {e}")
            return None, None

    # Merge past and future
    past_df = pd.concat(all_past.values(), axis=1)
    future_df = pd.concat(all_future.values(), axis=1)
    
    # Add time features to both
    for df in [past_df, future_df]:
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
    
    return past_df, future_df

# =========================================================
# 6. PREDICT NEXT 24 HOURS (WITH REAL FORECASTED WEATHER)
# =========================================================
def predict_next_24_hours(past_df, future_df):
    SEQ_LENGTH = 72
    last_72 = past_df.iloc[-SEQ_LENGTH:]
    
    # Start with the 72-hour sequence
    input_features_df = last_72[ALL_FEATURES].copy()
    for col in DEMAND_COLUMNS:
        input_features_df[col] = 0.0
    full_input_df = input_features_df[ALL_FEATURES + DEMAND_COLUMNS]
    input_scaled = scaler.transform(full_input_df)
    input_scaled_features = input_scaled[:, :len(ALL_FEATURES)]
    current_seq = input_scaled_features.reshape(1, SEQ_LENGTH, len(ALL_FEATURES))
    
    predictions = []
    
    # Loop through the next 24 hours of FORECASTED weather
    for i in range(24):
        # Predict the next hour based on current sequence
        pred = model.predict(current_seq, verbose=0)[0]
        predictions.append(pred)
        
        # Instead of copying the last weather, we use the REAL FORECASTED weather for the next hour
        # Get the forecast for the (i+1)-th future hour
        if i+1 < len(future_df):
            future_hour = future_df.iloc[i+1]
            
            # Create a new feature vector for the next step
            # We extract the weather features from the forecast
            new_weather = future_hour[ALL_FEATURES].values.reshape(1, -1)
            
            # Concatenate the weather features with the demand features (demand is not known yet, so we set to 0)
            # Actually, the scaler expects the 22 weather + 8 demands. 
            # We will just use the weather part, the model doesn't actually look at the demand columns as inputs during prediction.
            # However, to maintain the sequence shape, we will just swap the weather part.
            
            # Shift the sequence forward
            current_seq = np.roll(current_seq, -1, axis=1)
            # Replace the last hour's weather with the forecasted weather for the next hour
            current_seq[0, -1, :] = new_weather
            
        else:
            # If we don't have forecast data, fallback to the old method (copy last weather)
            new_step = current_seq[0, -1, :].copy()
            current_seq = np.roll(current_seq, -1, axis=1)
            current_seq[0, -1, :] = new_step
    
    return np.array(predictions)

# =========================================================
# 7. INVERSE TRANSFORM
# =========================================================
def inverse_demand(preds_scaled):
    dummy = np.zeros((len(preds_scaled), len(ALL_FEATURES) + 8))
    dummy[:, len(ALL_FEATURES):] = preds_scaled
    preds_real = scaler.inverse_transform(dummy)[:, len(ALL_FEATURES):]
    return preds_real

# =========================================================
# 8. DASHBOARD UI
# =========================================================
st.sidebar.header("⚙️ Control Panel")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 5 min", value=False) # Default off now

with st.spinner("🌤️ Fetching Historical & Forecast Weather data..."):
    past_df, future_df = fetch_weather_data()

if past_df is not None and future_df is not None:
    st.success("✅ Live Weather + 24hr Forecast loaded!")
    
    with st.spinner("🧠 Running AI forecast using real future weather..."):
        future_preds_scaled = predict_next_24_hours(past_df, future_df)
        future_preds_real = inverse_demand(future_preds_scaled)
    
    st.subheader("📊 Current Estimated Demand")
    cols = st.columns(4)
    for i, region in enumerate(region_order[:4]):
        with cols[i]:
            st.metric(
                label=region.replace("_", " "),
                value=f"{future_preds_real[0, i]:.0f} MW"
            )
    cols = st.columns(4)
    for i, region in enumerate(region_order[4:]):
        with cols[i]:
            st.metric(
                label=region.replace("_", " "),
                value=f"{future_preds_real[0, i+4]:.0f} MW"
            )
    
    st.subheader("📈 24-Hour Demand Forecast (Using Real Forecasted Weather)")
    now = datetime.now()
    future_times = [now + timedelta(hours=i) for i in range(24)]
    
    fig = go.Figure()
    for i, region in enumerate(region_order):
        fig.add_trace(go.Scatter(
            x=future_times,
            y=future_preds_real[:, i],
            mode='lines+markers',
            name=region.replace("_", " "),
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title="Predicted Demand per Region (Next 24 Hours)",
        xaxis_title="Time",
        yaxis_title="Demand (MW)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    with st.expander("📋 View Raw Forecast Data"):
        forecast_df = pd.DataFrame(
            future_preds_real,
            columns=region_order,
            index=future_times
        )
        st.dataframe(forecast_df.style.format("{:.0f}"))
    
    if auto_refresh:
        time.sleep(300)
        st.rerun()
else:
    st.error("❌ Could not fetch weather data. Check your internet connection.")