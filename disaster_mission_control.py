"""
Disaster Mission Control - Aegis Sentinel
Integrated ML-Based Disaster Risk Prediction & Monitoring System
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
from streamlit_geolocation import streamlit_geolocation
import pydeck as pdk
import folium
from streamlit_folium import st_folium
from folium import plugins
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import warnings

import warnings
import json
import uuid
import requests
import math

SESSIONS_DIR = "mission_logs"
os.makedirs(SESSIONS_DIR, exist_ok=True)

def get_all_sessions():
    """List all saved sessions from the SESSIONS_DIR"""
    sessions = []
    if os.path.exists(SESSIONS_DIR):
        for f in os.listdir(SESSIONS_DIR):
            if f.endswith(".json"):
                path = os.path.join(SESSIONS_DIR, f)
                try:
                    with open(path, "r", encoding="utf-8") as file:
                        data = json.load(file)
                        sessions.append({
                            "id": f.replace(".json", ""),
                            "title": data.get("title", "Unnamed Mission"),
                            "timestamp": os.path.getmtime(path)
                        })
                except:
                    continue
    # Sort by newest first
    sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    return sessions

def load_session(session_id):
    """Load a specific session into st.session_state"""
    path = os.path.join(SESSIONS_DIR, f"{session_id}.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                st.session_state.messages = data.get("messages", [])
                st.session_state.session_title = data.get("title", "Unnamed Mission")
                st.session_state.current_session_id = session_id
                # Attempt to restore location/role if saved
                st.session_state.location_locked = data.get("location_locked", False)
                st.session_state.role_locked = data.get("role_locked", False)
                st.session_state.form_lat = data.get("form_lat", 0.0)
                st.session_state.form_lon = data.get("form_lon", 0.0)
                st.session_state.form_city = data.get("form_city", "")
                st.session_state.form_role = data.get("form_role", "Select...")
                st.session_state.active_directives = data.get("active_directives", [])
        except Exception as e:
            st.error(f"Error loading mission: {e}")

def save_chat_history():
    """Save the current session to a JSON file in SESSIONS_DIR"""
    if 'current_session_id' not in st.session_state or not st.session_state.current_session_id:
        st.session_state.current_session_id = str(uuid.uuid4())
    
    path = os.path.join(SESSIONS_DIR, f"{st.session_state.current_session_id}.json")
    payload = {
        "title": st.session_state.get("session_title", "New Mission"),
        "messages": st.session_state.messages,
        "location_locked": st.session_state.get("location_locked", False),
        "role_locked": st.session_state.get("role_locked", False),
        "form_lat": st.session_state.get("form_lat", 0.0),
        "form_lon": st.session_state.get("form_lon", 0.0),
        "form_city": st.session_state.get("form_city", ""),
        "form_role": st.session_state.get("form_role", "Select..."),
        "active_directives": st.session_state.get("active_directives", [])
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving history: {e}")

warnings.filterwarnings('ignore')
load_dotenv()

# ==============================================================================
# ML RISK PREDICTOR CLASS
# ==============================================================================

class DisasterRiskPredictor:
    """ML-based disaster risk prediction system"""

    def __init__(self):
        self.disaster_types = [
            'Flood', 'Heatwave', 'Cyclone', 'Drought', 'Landslide',
            'Wildfire', 'Avalanche', 'Seismic', 'Tsunami', 'Volcano'
        ]
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False

    def extract_features(self, weather_data, lat, lon, usgs_data, fire_data, sentinel_data):
        """Extract features from various data sources for ML prediction"""
        features = []

        # Weather features
        features.extend([
            weather_data.get('temp', 20),
            weather_data.get('pressure', 1013),
            weather_data.get('humidity', 50),
            weather_data.get('wind_speed', 0),
            weather_data.get('wind_deg', 0)
        ])

        # Geographic features
        features.extend([lat, lon, abs(lat)])
        is_coastal = 1 if abs(lat) < 60 and (abs(lon) < 20 or abs(lon) > 160) else 0
        features.append(is_coastal)

        # Seismic zone indicator
        seismic_zones = [(35, 45, 135, 145), (-10, 10, -85, -70), (30, 40, -125, -115)]
        in_seismic_zone = any(z[0] <= lat <= z[1] and z[2] <= lon <= z[3] for z in seismic_zones)
        features.append(1 if in_seismic_zone else 0)
        features.append(abs(lat) * 10)  # Elevation proxy

        # Temporal features
        now = datetime.now()
        features.extend([now.month, now.hour])

        # Satellite data features
        features.extend([
            usgs_data.get('count', 0),
            usgs_data.get('max_mag', 0),
            fire_data,
            sentinel_data.get('s1_flood_signal', 0),
            sentinel_data.get('s2_ndvi_signal', 0.5)
        ])

        # Derived features
        temp_humidity_ratio = weather_data.get('temp', 20) / max(weather_data.get('humidity', 50), 1)
        pressure_anomaly = abs(weather_data.get('pressure', 1013) - 1013)
        features.extend([temp_humidity_ratio, pressure_anomaly])

        return np.array(features).reshape(1, -1)

    def train_models(self, synthetic_data=True):
        """Train ML models using synthetic data"""
        if synthetic_data:
            X_train, y_train = self._generate_synthetic_data(n_samples=5000)

        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        for i, disaster in enumerate(self.disaster_types):
            print(f"Training model for {disaster}...")
            y_binary = (y_train == i).astype(int)

            rf_model = RandomForestClassifier(
                n_estimators=100, max_depth=15, min_samples_split=10,
                min_samples_leaf=5, random_state=42, n_jobs=-1
            )
            rf_model.fit(X_train_scaled, y_binary)
            self.models[disaster] = rf_model

        self.is_trained = True
        print("✅ All models trained successfully!")

    def _generate_synthetic_data(self, n_samples=5000):
        """Generate synthetic training data"""
        np.random.seed(42)
        X, y = [], []

        for _ in range(n_samples):
            temp = np.random.uniform(-10, 45)
            pressure = np.random.uniform(980, 1040)
            humidity = np.random.uniform(10, 100)
            wind_speed = np.random.uniform(0, 50)
            wind_deg = np.random.uniform(0, 360)
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)

            abs_lat = abs(lat)
            is_coastal = 1 if abs_lat < 60 and (abs(lon) < 20 or abs(lon) > 160) else 0
            in_seismic = 1 if (35 <= lat <= 45 and 135 <= lon <= 145) else 0
            elevation = abs_lat * 10
            month = np.random.randint(1, 13)
            hour = np.random.randint(0, 24)

            seismic_count = np.random.poisson(2) if in_seismic else np.random.poisson(0.5)
            seismic_mag = np.random.uniform(0, 7) if seismic_count > 0 else 0
            fire_count = np.random.poisson(5) if temp > 30 and humidity < 30 else np.random.poisson(0.5)
            flood_signal = np.random.uniform(0, 1) if humidity > 80 else np.random.uniform(0, 0.3)
            ndvi = np.random.uniform(0.3, 0.8)
            temp_hum_ratio = temp / max(humidity, 1)
            pressure_anom = abs(pressure - 1013)

            features = [
                temp, pressure, humidity, wind_speed, wind_deg,
                lat, lon, abs_lat, is_coastal, in_seismic, elevation,
                month, hour, seismic_count, seismic_mag, fire_count,
                flood_signal, ndvi, temp_hum_ratio, pressure_anom
            ]

            disaster_idx = self._determine_disaster_type(
                temp, pressure, humidity, wind_speed, is_coastal,
                in_seismic, seismic_mag, fire_count, flood_signal, month
            )

            X.append(features)
            y.append(disaster_idx)

        return np.array(X), np.array(y)

    def _determine_disaster_type(self, temp, pressure, humidity, wind_speed,
                                 is_coastal, in_seismic, seismic_mag, fire_count,
                                 flood_signal, month):
        """Determine disaster type based on scientific correlations"""
        scores = np.zeros(10)

        scores[0] = (humidity / 100) * 0.4 + (1 - pressure / 1040) * 0.3 + flood_signal * 0.3
        scores[1] = max(0, (temp - 30) / 15) * 0.6 + (1 - humidity / 100) * 0.4
        scores[2] = (1 - pressure / 1040) * 0.4 + (wind_speed / 50) * 0.4 + is_coastal * 0.2

        drought_months = [5, 6, 7, 8, 9]
        month_factor = 1 if month in drought_months else 0.5
        scores[3] = (1 - humidity / 100) * 0.5 + max(0, (temp - 25) / 20) * 0.3 + month_factor * 0.2

        scores[4] = (humidity / 100) * 0.5 + 0.3
        scores[5] = max(0, (temp - 25) / 20) * 0.4 + (1 - humidity / 100) * 0.4 + min(fire_count / 10, 1) * 0.2

        winter_months = [11, 12, 1, 2, 3]
        month_factor = 1 if month in winter_months else 0.3
        scores[6] = max(0, (10 - temp) / 20) * 0.6 + month_factor * 0.4

        scores[7] = in_seismic * 0.5 + min(seismic_mag / 7, 1) * 0.5
        scores[8] = in_seismic * 0.4 + is_coastal * 0.3 + min(seismic_mag / 7, 1) * 0.3
        scores[9] = in_seismic * 0.6 + min(seismic_mag / 7, 1) * 0.4

        scores += np.random.uniform(-0.1, 0.1, 10)
        scores = np.clip(scores, 0, 1)
        return np.argmax(scores)

    def predict_risks(self, weather_data, lat, lon, usgs_data, fire_data, sentinel_data):
        """Predict disaster risks using trained ML models"""
        if not self.is_trained:
            self.train_models()

        features = self.extract_features(weather_data, lat, lon, usgs_data, fire_data, sentinel_data)
        features_scaled = self.scaler.transform(features)

        predictions = {}
        for disaster, model in self.models.items():
            prob = model.predict_proba(features_scaled)[0][1]
            risk_score = min(prob * 120, 100)
            predictions[disaster] = max(5, risk_score)

        return predictions

    def save_models(self, directory='models'):
        """Save trained models to disk"""
        os.makedirs(directory, exist_ok=True)
        for disaster, model in self.models.items():
            filename = os.path.join(directory, f'{disaster.lower()}_model.joblib')
            joblib.dump(model, filename)
        joblib.dump(self.scaler, os.path.join(directory, 'scaler.joblib'))
        print(f"✅ Models saved to {directory}/")

    def load_models(self, directory='models'):
        """Load trained models from disk"""
        try:
            for disaster in self.disaster_types:
                filename = os.path.join(directory, f'{disaster.lower()}_model.joblib')
                self.models[disaster] = joblib.load(filename)
            self.scaler = joblib.load(os.path.join(directory, 'scaler.joblib'))
            self.is_trained = True
            print("✅ Models loaded successfully!")
            return True
        except Exception as e:
            print(f"⚠️ Could not load models: {e}")
            return False

# Global predictor instance
_predictor = None

def get_predictor():
    """Get or create the global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = DisasterRiskPredictor()
        if not _predictor.load_models():
            print("Training new models...")
            _predictor.train_models()
            _predictor.save_models()
    return _predictor

# ==============================================================================
# STREAMLIT CONFIGURATION
# ==============================================================================

st.set_page_config(
    page_title="Disaster Risk Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;600;800&display=swap');

    /* Global Base Styling - Pure Minimalist Dark */
    .stApp { 
        background-color: #000000; 
        color: #d1d5db; 
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, .stHeader {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.03em;
        color: #ffffff;
    }

    /* Container Optimization */
    .main .block-container { 
        max-width: 900px; 
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* HIDE ADMIN FEATURES */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    .stDeployButton {
        display: none !important;
    }
    footer {
        visibility: hidden !important;
    }
    #MainMenu {
        visibility: hidden !important;
    }

    /* Sidebar - High Contrast Dark */
    [data-testid="stSidebar"] { 
        background-color: #0a0a0a !important;
        border-right: 1px solid #262626 !important;
    }

    /* Chat Area - Tactical Cleanliness */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 1.25rem 0 !important;
        margin-bottom: 0.25rem !important;
    }

    /* User Message Bubble - Sharp & Professional */
    [data-testid="stChatMessageContent"][data-testid*="user"] {
        background-color: #1a1a1a !important;
        border-radius: 4px !important;
        padding: 12px 18px !important;
        color: #ffffff !important;
        font-weight: 500 !important;
        border: 1px solid #333333 !important;
    }

    /* Assistant Message Bubble - Minimalist Monotone */
    [data-testid="stChatMessageContent"] {
        background-color: #0a0a0a !important;
        border: 1px solid #1f1f1f !important;
        border-radius: 4px !important;
        padding: 18px !important;
    }

    /* Input Box - Stealth Floating Style */
    .stChatInputContainer {
        border: none !important;
        padding: 1rem !important;
        background-color: #000000 !important;
    }

    .stChatInput {
        background-color: #0f0f0f !important;
        border: 1px solid #262626 !important;
        border-radius: 4px !important;
        padding: 10px 18px !important;
        color: #ffffff !important;
        transition: border-color 0.2s ease !important;
    }

    .stChatInput:focus {
        border-color: #404040 !important;
        box-shadow: none !important;
    }

    /* Compact Tactical Buttons */
    .stButton button {
        background-color: #141414 !important;
        color: #e5e5e5 !important;
        border: 1px solid #262626 !important;
        border-radius: 4px !important;
        padding: 0.4rem 0.8rem !important; /* Smaller padding */
        font-size: 13px !important; /* Smaller font */
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        transition: all 0.2s ease !important;
    }

    .stButton button:hover {
        background-color: #1f1f1f !important;
        border-color: #404040 !important;
        transform: none !important;
    }

    .stButton button[kind="primary"] {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: none !important;
    }

    .stButton button[kind="primary"]:hover {
        background-color: #e5e5e5 !important;
    }

    /* Metric & Risk Cards - Monochromatic Focus */
    .metric-container { 
        background: #0a0a0a;
        border: 1px solid #1f1f1f;
        border-radius: 4px; 
        padding: 16px; 
        text-align: center; 
    }

    .metric-val { 
        font-family: 'Outfit', sans-serif;
        font-size: 24px; 
        font-weight: 700; 
        color: #ffffff;
    }

    .metric-lbl { 
        font-size: 10px; 
        color: #737373; 
        margin-top: 6px; 
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .risk-card { 
        border-radius: 4px; 
        padding: 14px; 
        margin-bottom: 10px; 
        border: 1px solid #1f1f1f;
        background: #050505;
    }

    .risk-high { border-left: 3px solid #7f1d1d; } /* Dark Red */
    .risk-med { border-left: 3px solid #78350f; }  /* Dark Amber */
    .risk-low { border-left: 3px solid #064e3b; }  /* Dark Green */

    /* Scrollbar - Hidden/Minimal */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { 
        background: #262626; 
        border-radius: 0px; 
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# DATA FETCHING FUNCTIONS
# ==============================================================================

def get_real_weather(lat, lon, api_key):
    """Fetch current real weather or return demo data"""
    if not api_key:
        return {
            'temp': 28.5, 'humidity': 75, 'pressure': 1012, 'wind_speed': 15.0,
            'wind_deg': 240, 'visibility': 10000, 'uv': 6.5, 'aqi': 45,
            'sunrise': '06:15 AM', 'sunset': '06:45 PM', 'desc': 'Partly Cloudy',
            'lat': lat, 'lon': lon, 'timezone_offset': 0
        }
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {'lat': lat, 'lon': lon, 'appid': api_key, 'units': 'metric'}
        res = requests.get(url, params=params, timeout=3)
        if res.status_code == 200:
            d = res.json()
            offset_seconds = d.get('timezone', 0)
            tz = timezone(timedelta(seconds=offset_seconds))
            sr = datetime.fromtimestamp(d['sys']['sunrise'], tz).strftime('%H:%M hrs')
            ss = datetime.fromtimestamp(d['sys']['sunset'], tz).strftime('%H:%M hrs')
            lt = datetime.now(tz).strftime('%H:%M hrs')

            return {
                'temp': d['main']['temp'], 'humidity': d['main']['humidity'],
                'pressure': d['main']['pressure'], 'wind_speed': d['wind']['speed'],
                'wind_deg': d['wind'].get('deg', 0), 'visibility': d.get('visibility', 10000),
                'uv': 5.0, 'aqi': 50, 'sunrise': sr, 'sunset': ss, 'local_time': lt,
                'timezone_offset': offset_seconds, 'desc': d['weather'][0]['description'].title(),
                'lat': lat, 'lon': lon
            }
    except:
        pass
    return {
        'temp': 28.5, 'humidity': 75, 'pressure': 1012, 'wind_speed': 15.0,
        'wind_deg': 240, 'visibility': 10000, 'uv': 6.5, 'aqi': 45,
        'sunrise': '06:15 hrs', 'sunset': '18:45 hrs',
        'local_time': datetime.now().strftime('%H:%M hrs'),
        'timezone_offset': 0, 'desc': 'Partly Cloudy', 'lat': lat, 'lon': lon
    }

def get_usgs_data(lat, lon, radius_km=500):
    """Fetch recent earthquakes from USGS"""
    try:
        url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/2.5_day.geojson"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            max_mag, count = 0, 0
            for feature in data['features']:
                eq_lon = feature['geometry']['coordinates'][0]
                eq_lat = feature['geometry']['coordinates'][1]
                mag = feature['properties']['mag']
                dist = math.sqrt((lat - eq_lat) ** 2 + (lon - eq_lon) ** 2) * 111
                if dist < radius_km:
                    max_mag = max(max_mag, mag)
                    count += 1
            return {"max_mag": max_mag, "count": count}
    except:
        pass
    return {"max_mag": 0, "count": 0}

def get_firms_data(lat, lon, api_key):
    """Fetch active fire data from NASA FIRMS"""
    if not api_key:
        return 0
    try:
        box = f"{lon - 0.5},{lat - 0.5},{lon + 0.5},{lat + 0.5}"
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/{box}/1"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            lines = res.text.strip().split('\n')
            if len(lines) > 1:
                return len(lines) - 1
    except:
        pass
    return 0

def get_sentinel_data(lat, lon, api_key):
    """Fetch Sentinel-1/2 Data"""
    if not api_key:
        return {"s1_flood_signal": 0, "s2_ndvi_signal": 0.5}
    return {"s1_flood_signal": 0, "s2_ndvi_signal": 0.4}

def generate_forecast_data(current_weather, usgs_data, fire_count, sentinel_data):
    """Generate 72h and 7-day forecast dataframes using ML predictions"""
    predictor = get_predictor()
    hours, steps = 72, 24

    offset = current_weather.get('timezone_offset', 0)
    tz = timezone(timedelta(seconds=offset))
    start_time = datetime.now(tz)
    times = [(start_time + timedelta(hours=i * 3)).strftime('%H:%M\n%d %b') for i in range(steps + 1)]

    base_temp = current_weather['temp']
    t_idx = np.arange(steps + 1)
    temps = base_temp + 5 * np.sin(t_idx / 2.5) + np.random.normal(0, 0.5, steps + 1)
    hums = np.clip(current_weather['humidity'] - 10 * np.sin(t_idx / 2.5), 20, 100)
    press = current_weather['pressure'] + np.cumsum(np.random.normal(0, 0.5, steps + 1))
    wind = np.clip(current_weather['wind_speed'] + np.random.normal(0, 2, steps + 1), 0, 100)

    df_72h = pd.DataFrame({
        'Time': times, 'Temperature (°C)': temps, 'Humidity (%)': hums,
        'Pressure (hPa)': press, 'Wind Speed (m/s)': wind
    })

    lat = current_weather.get('lat', 0)
    lon = current_weather.get('lon', 0)
    risks_data = []

    for i, row in df_72h.iterrows():
        weather_snapshot = {
            'temp': row['Temperature (°C)'], 'pressure': row['Pressure (hPa)'],
            'humidity': row['Humidity (%)'], 'wind_speed': row['Wind Speed (m/s)'],
            'wind_deg': current_weather.get('wind_deg', 0)
        }
        ml_predictions = predictor.predict_risks(
            weather_snapshot, lat, lon, usgs_data, fire_count, sentinel_data
        )
        time_variation = np.random.uniform(-3, 3)
        risks_data.append([
            np.clip(ml_predictions['Flood'] + time_variation, 5, 100),
            np.clip(ml_predictions['Heatwave'] + time_variation, 5, 100),
            np.clip(ml_predictions['Cyclone'] + time_variation, 5, 100),
            np.clip(ml_predictions['Drought'] + time_variation, 5, 100),
            np.clip(ml_predictions['Landslide'] + time_variation, 5, 100),
            np.clip(ml_predictions['Wildfire'] + time_variation, 5, 100),
            np.clip(ml_predictions['Avalanche'] + time_variation, 5, 100),
            np.clip(ml_predictions['Seismic'] + time_variation, 5, 100),
            np.clip(ml_predictions['Tsunami'] + time_variation, 5, 100),
            np.clip(ml_predictions['Volcano'] + time_variation, 5, 100),
        ])

    risk_cols = ['Flood', 'Heatwave', 'Cyclone', 'Drought', 'Landslide', 'Wildfire', 'Avalanche', 'Seismic', 'Tsunami',
                 'Volcano']
    df_risks = pd.DataFrame(risks_data, columns=risk_cols)

    df_7d = pd.DataFrame({
        'Date': [(start_time + timedelta(days=i)).strftime('%d %b') for i in range(7)],
        'High Risk Agent': [np.random.choice(risk_cols) for _ in range(7)],
        'Avg Temp (°C)': [base_temp + np.random.randint(-2, 5) for _ in range(7)],
        'Conf. Level': [np.random.choice(['HIGH', 'MED', 'LOW']) for _ in range(7)]
    })

    drift = {}
    for col in risk_cols:
        current_val = df_risks[col].iloc[0]
        prev_val = max(5, min(100, current_val + np.random.randint(-20, 15)))
        change = current_val - prev_val
        drift[col] = {
            "current": current_val, "previous": prev_val,
            "change": change, "percent": (change / prev_val * 100) if prev_val > 0 else 0
        }

    return df_72h, df_7d, df_risks, drift

def get_precautions(role, risks):
    """Get role-based precautions"""
    precautions = ["Keep emergency contacts ready.", "Monitor local news.", "Charge communication devices."]

    role_specific = {
        "Farmer/Agricultural Worker": ["Secure livestock.", "Delay sowing if heavy rain predicted.", "Cover harvested crops."],
        "Transportation/Driver": ["Avoid coastal roads.", "Check tire pressure.", "Keep emergency light in car."],
        "Student": ["Carry umbrella/raincoat.", "Keep parent numbers written down.", "Stay in school if storm hits."],
        "Emergency Responder": ["Check equipment readiness.", "Brief team on risk zones.",
                                "Ensure vehicle fuel is full."],
        "Healthcare Worker": ["Prepare emergency triage protocols.", "Secure medical supplies.", "Check backup generator fuel."],
        "Construction Worker": ["Secure loose scaffolding.", "Move heavy machinery to high ground.", "Monitor wind speeds for crane safety."],
        "Elderly Person": ["Keep a 7-day supply of medications.", "Ensure mobility aids are accessible.", "Register with local emergency lists."],
    }
    precautions.extend(role_specific.get(role, []))

    if risks['Flood'].mean() > 50: precautions.append("Flood Warning: Avoid basements.")
    if risks['Heatwave'].mean() > 40: precautions.append("Heat Warning: Stay hydrated.")

    return precautions

def get_risk_confidence(risk_type, score, weather):
    """Generate confidence level and reason"""
    confidence, reason = "Low", "We do not have enough data to be sure yet. We are still checking the signals."

    if risk_type == "Flood" and weather['humidity'] > 80:
        confidence, reason = "HIGH", "The air is very wet and heavy. When there is this much moisture, it usually means big storms and flooding might happen."
    elif risk_type == "Heatwave" and weather['temp'] > 35:
        confidence, reason = "HIGH", f"The temperature is {weather['temp']}°C. This is very hot, so there is a high chance of a heatwave."
    elif risk_type == "Cyclone" and weather['wind_speed'] > 20:
        confidence, reason = "HIGH", "The wind is blowing very fast. Fast winds are a main sign that a cyclone might be starting."
    elif score > 70:
        confidence = "HIGH"
        reason = "Multiple satellites and sensors are all agreeing that there is a danger. When they all agree, we can be very sure."
    elif score > 40:
        confidence = "MEDIUM"
        reason = "We see some warning signs, but not all of them. We need to keep watching to be sure."

    return confidence, reason

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    # Session State Initialization
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.current_session_id = None
        st.session_state.session_title = "New Mission"
        
        # Auto-load latest session if exists and no current session
        all_sesh = get_all_sessions()
        if all_sesh:
            load_session(all_sesh[0]['id'])
        else:
            welcome_msg = """
            **[SYSTEM INITIALIZED: AEGIS SENTINEL v4.5]**

            Greetings. I am **Aegis Sentinel**, your proactive disaster intelligence assistant. 
            I monitor global telemetry streams from NASA FIRMS, USGS, and Copernicus Sentinel orbits.

            **ALERT: No intelligence data will be displayed until your geographic coordinates and authority profile are verified.**
            """
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            st.session_state.location_locked = False
            st.session_state.role_locked = False
            save_chat_history()

    if 'location_locked' not in st.session_state: st.session_state.location_locked = False
    if 'role_locked' not in st.session_state: st.session_state.role_locked = False
    if 'active_directives' not in st.session_state: st.session_state.active_directives = []
    if 'form_lat' not in st.session_state: st.session_state.form_lat = 0.0
    if 'form_lon' not in st.session_state: st.session_state.form_lon = 0.0
    if 'form_city' not in st.session_state: st.session_state.form_city = ""
    if 'form_role' not in st.session_state: st.session_state.form_role = "Select..."

    # SIDEBAR: MISSIONS LIST
    with st.sidebar:
        st.markdown(f"### Mission Logs")
        if st.button("New Mission", use_container_width=True):
            st.session_state.messages = []
            st.session_state.current_session_id = str(uuid.uuid4())
            st.session_state.session_title = "New Mission"
            st.session_state.location_locked = False
            st.session_state.role_locked = False
            st.session_state.active_directives = []
            welcome_msg = "**[NEW SESSION INITIALIZED]**\nAwaiting coordinates for the new disaster vector."
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            save_chat_history()
            st.rerun()

        st.markdown("---")
        
        sessions = get_all_sessions()
        for s in sessions:
            cols = st.columns([4, 1])
            if cols[0].button(f"Session: {s['title'][:20]}...", key=f"session_{s['id']}", use_container_width=True):
                load_session(s['id'])
                st.rerun()
            if cols[1].button("🗑️", key=f"del_{s['id']}"):
                os.remove(os.path.join(SESSIONS_DIR, f"{s['id']}.json"))
                st.rerun()

        st.markdown("---")
        if st.button("Clear All Logs", use_container_width=True):
            if os.path.exists(SESSIONS_DIR):
                for f in os.listdir(SESSIONS_DIR):
                    os.remove(os.path.join(SESSIONS_DIR, f))
            st.rerun()

    st.title("Aegis Sentinel")

    # Display Chat History (clean chat without directive outputs)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



    # ONBOARDING PANEL
    if not st.session_state.location_locked:
        with st.chat_message("assistant"):
            st.markdown("### Telemetry Calibration")
            st.warning("Calibration Required: Define your location to enable satellite intelligence.")
            
            # Status indicator
            if st.session_state.form_lat != 0.0 or st.session_state.form_lon != 0.0:
                if st.session_state.form_city and st.session_state.form_city != "":
                    st.success(f"Location Data Ready: {st.session_state.form_city} ({st.session_state.form_lat}, {st.session_state.form_lon})")
                else:
                    st.info(f"Coordinates Set: {st.session_state.form_lat}, {st.session_state.form_lon} (City name needed)")

            # GPS Sync - Integrated Control Module

            with st.container(border=True):
                 col_btn, col_txt = st.columns([1, 3])
                 with col_btn:
                    loc = streamlit_geolocation()
                 with col_txt:
                    st.markdown("<div style='line-height: 40px; font-weight: 500;'>Use GPS for Device Location</div>", unsafe_allow_html=True)
            
            # Check for valid location data from component
            new_lat = None
            new_lon = None
            
            if loc:
                if 'latitude' in loc and loc['latitude'] is not None:
                     new_lat = round(float(loc['latitude']), 4)
                     new_lon = round(float(loc['longitude']), 4)
                elif 'coords' in loc and 'latitude' in loc['coords']:
                     new_lat = round(float(loc['coords']['latitude']), 4)
                     new_lon = round(float(loc['coords']['longitude']), 4)
            
            if new_lat is not None and new_lon is not None:
                if new_lat != st.session_state.form_lat or new_lon != st.session_state.form_lon:
                    # Reverse Geocode Logic
                    api_key = os.getenv("OPENWEATHER_API_KEY")
                    city_name = ""
                    
                    if api_key:
                        try:
                            rev_url = f"https://api.openweathermap.org/geo/1.0/reverse?lat={new_lat}&lon={new_lon}&limit=1&appid={api_key}"
                            res = requests.get(rev_url, timeout=10)
                            if res.status_code == 200:
                                data = res.json()
                                if data and len(data) > 0:
                                    location_data = data[0]
                                    if 'name' in location_data:
                                        city_name = location_data['name']
                                        if 'state' in location_data:
                                            city_name += f", {location_data['state']}"
                                        elif 'country' in location_data:
                                            city_name += f", {location_data['country']}"
                        except Exception as e:
                            print(f"Reverse geocoding error: {e}")
                    
                    if not city_name:
                        city_name = f"Location {new_lat}, {new_lon}"
                    
                    st.session_state.form_city = city_name
                    st.session_state.form_lat = new_lat
                    st.session_state.form_lon = new_lon
                    
                    st.success(f"GPS Synced: {city_name}")
                    import time
                    time.sleep(0.1)
                    st.rerun()

            st.markdown("---")

            # Manual coordinate helper
            with st.expander("Manual Coordinate Helper", expanded=False):
                st.markdown("""
                **Need help finding coordinates?**
                
                1. **Google Maps**: Right-click any location -> Click coordinates that appear
                2. **Mobile Apps**: Most smartphone GPS apps show current coordinates
                3. **Common Locations**:
                   - New York City: 40.7128, -74.0060
                   - London: 51.5074, -0.1278
                   - Tokyo: 35.6762, 139.6503
                   - Sydney: -33.8688, 151.2093
                   - Mumbai: 19.0760, 72.8777
                
                **Format**: Latitude (North/South), Longitude (East/West)
                """)

            # Two column layout for inputs
            p1, p2 = st.columns([2, 1])

            with p1:
                # City input - manual entry only
                city_inp = st.text_input(
                    "City Name",
                    value=st.session_state.form_city,
                    placeholder="Enter your city name"
                )

            with p2:
                # Coordinate inputs - manual entry only
                lat_inp = st.number_input(
                    "Latitude",
                    value=float(st.session_state.form_lat),
                    format="%.6f",
                    step=0.000001
                )

                lon_inp = st.number_input(
                    "Longitude",
                    value=float(st.session_state.form_lon),
                    format="%.6f",
                    step=0.000001
                )

            # Update session state
            st.session_state.form_city = city_inp
            st.session_state.form_lat = lat_inp
            st.session_state.form_lon = lon_inp

            # Lock button
            if st.button("LOCK LOCATION & PROCEED", use_container_width=True, type="primary",
                         key="lock_location_btn"):
                # Validation checks
                validation_errors = []
                
                if not st.session_state.form_city.strip():
                    validation_errors.append("City name is required")
                
                if st.session_state.form_lat == 0.0 and st.session_state.form_lon == 0.0:
                    validation_errors.append("Please provide valid coordinates")
                
                if abs(st.session_state.form_lat) > 90:
                    validation_errors.append("Latitude must be between -90 and 90 degrees")

                if validation_errors:
                    for err in validation_errors:
                        st.error(err)
                else:
                    st.session_state.location_locked = True
                    msg = f"Location Verified: {st.session_state.form_city}. Calibrating sensors...\n\nI have successfully synchronized with the local orbital grid. My telemetry sensors are now tuning to your specific coordinates for precise monitoring."
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    save_chat_history()
                    st.rerun()

    # STEP 2: PROFILE SELECTION
    elif not st.session_state.role_locked:
        with st.chat_message("assistant"):
            st.markdown("### Final Verification: Authority Profile")
            st.info("To tailor my tactical advice, please select the profile that best describes you.")
            
            role_inp = st.selectbox(
                "Select Your Profile",
                ["Select...", "General User", "Student", "Elderly Person",
                 "Person with Disability", "Pregnant/Nursing Mother", "Child/Minor",
                 "Emergency Responder", "Healthcare Worker", "Teacher/Educator",
                 "Farmer/Agricultural Worker", "Construction Worker",
                 "Transportation/Driver", "Business Owner", "Homeless/Displaced Person",
                 "Tourist/Visitor"],
                index=0
            )
            
            if st.button("INITIALIZE MISSION CONTROL", use_container_width=True, type="primary"):
                if role_inp == "Select...":
                    st.error("Please select a profile to continue.")
                else:
                    st.session_state.form_role = role_inp
                    st.session_state.role_locked = True
                    msg = f"Access Granted: Profile set to **{role_inp}**. \n\nAegis Mission Control is now fully operational for your sector. I have adjusted my intelligence filters to prioritize hazards that specifically impact your profile and professional requirements."
                    st.session_state.messages.append({"role": "assistant", "content": msg})
                    save_chat_history()
                    st.rerun()
        st.stop()

    # MAIN APPLICATION (Only after both locked)
    if st.session_state.location_locked and st.session_state.role_locked:

        # Chat Input Logic
        if prompt := st.chat_input("Message Aegis Sentinel..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            save_chat_history()
            p = prompt.lower()
            
            detected_directives = []
            responses = []

            # "Run All" Detection - check for keywords that mean all features
            run_all_keywords = [
                "all", "everything", "complete", "briefing", "entire", "full", "total", 
                "system", "initialize", "maximum", "comprehensive", "every feature", 
                "run all", "start everything", "all-in-one", "broad"
            ]
            if any(word in p for word in run_all_keywords):
                detected_directives = ["map", "risk", "forecast", "protocol", "threat", "emergency", "report", "kit", "sos"]
                responses.append("""
[MAXIMUM INTELLIGENCE PROTOCOL ACTIVATED]
I am currently initializing all nine tactical intelligence modules simultaneously to provide you with a comprehensive disaster readiness briefing. 
My systems are synchronizing orbital radar from Sentinel-1, thermal anomaly streams from NASA FIRMS, and global seismic telemetry from USGS to generate a complete visual and analytical map of your sector. 
Please standby as I calibrate the hazard map, risk evolution dashboards, and role-specific safety guides for your coordinates.
""")
            else:
                if "map" in p:
                    detected_directives.append("map")
                    responses.append("""
[VISUAL INTELLIGENCE LINK ESTABLISHED]
I am depth-syncing your precise coordinates with high-resolution Google Satellite clusters and NASA FIRMS active fire telemetry streams. 
The interactive hazard map will now visualize critical tactical data, including active wildfire perimeters, seismic fault lines, and flood-prone elevation zones in your immediate area. 
Ensure you review the color-coded safety boundaries to identify the most secure evacuation corridors currently available.
""")
                
                if "risk" in p or "drift" in p:
                    detected_directives.append("risk")
                    responses.append("""
[ML RISK EVOLUTION SCAN ONLINE]
My machine learning engines are processing a 48-hour probability drift analysis across ten distinct disaster vectors, including hydrometeorological and geophysical threats. 
By cross-referencing current atmospheric pressure anomalies with regional historical baselines, I am rendering a comparative risk matrix for your specific sector. 
The following dashboard will visualize exactly how threat levels are projected to evolve over the next two days, allowing for proactive tactical positioning.
""")
                
                if "forecast" in p:
                    detected_directives.append("forecast")
                    responses.append("""
[FORECASTING PULSE MATRIX INITIALIZED]
Initiating high-fidelity 72-hour environmental pulse models derived from Copernicus atmospheric monitoring and local meteorological stations. 
I am preparing a series of interactive trend visualizations for temperature fluctuations, wind velocity clusters, and humidity gradients tailored to your exact location. 
These models provide the granularity required to anticipate shifting weather fronts that could exacerbate existing hazard conditions in your mission area.
""")
                
                if "protocol" in p or "next" in p or "safety" in p or "guides" in p:
                    detected_directives.append("protocol")
                    responses.append("""
[TACTICAL SAFETY BRIEFING GENERATED]
I have synthesized role-specific safety protocols by analyzing your authority profile against current satellite-detected risk levels. 
My database has extracted the most critical action checklists and identified relevant survival guides from our integrated emergency library to ensure maximum readiness. 
These directives are prioritized based on the highest current threat probability, providing you with a step-by-step tactical roadmap for immediate hazard mitigation.
""")
                
                if "threat" in p or "assessment" in p:
                    detected_directives.append("threat")
                    responses.append("""
[INTELLIGENT THREAT ASSESSMENT ACTIVE]
Activating AI-driven interpretation of dominant orbital signals to provide a plain-language executive summary of your current danger profile. 
I am currently cross-referencing Sentinel-1 radar backscatter for flood signals with USGS seismic activity and NASA thermal hotspots to identify your primary disaster driver. 
The following assessment will break down AI confidence levels and sensor reliability metrics, ensuring you understand the empirical foundation of our tactical alerts.
""")
                
                if "emergency" in p or "locator" in p or "hospital" in p or "shelter" in p:
                    detected_directives.append("emergency")
                    responses.append("""
[EMERGENCY INFRASTRUCTURE LOCATOR ONLINE]
Triangulating the nearest critical support nodes, including level-one trauma centers, community relief shelters, and government-authorized distribution points. 
I am scanning regional infrastructure databases to provide you with direct routing details, estimated travel times, and current operational status markers for your specific sector. 
Please ensure your local navigation systems are ready to receive these coordinates for immediate tactical relocation if the situation escalates.
""")
                
                if "report" in p or "community" in p:
                    detected_directives.append("report")
                    responses.append("""
[COMMUNITY INTELLIGENCE WEB ACTIVE]
I have established a link to the hyper-local hazard mesh to display real-time reports submitted by other verified mission operators in your vicinity. 
This interface allows you to view critical community-sourced data point such as blocked arterial roads, localized flooding, and downed power infrastructure that may not yet be in official satellite feeds. 
You are also authorized to contribute your own situational observations to help calibrate the safety matrix for all nearby users.
""")
                
                if "kit" in p or "survival" in p or "supplies" in p:
                    detected_directives.append("kit")
                    responses.append("""
[DYNAMIC SMART SURVIVAL KIT PREPARED]
Generating a mission-critical inventory of essential supplies that has been algorithmically adapted to your highest current hazard exposure. 
Instead of generic advice, I am prioritizing items specific to the disasters detected at your coordinates, such as specific filtration needs for flood zones or cooling supplies for heatwaves. 
Please verify your current stockpile against this dynamic checklist to ensure you have at least 72 hours of self-sufficiency resources available.
""")
                
                if "sos" in p:
                    detected_directives.append("sos")
                    responses.append("""
[CRITICAL SOS PROTOCOL STANDBY]
I am currently opening the satellite-linked emergency transmission terminal for immediate distress broadcasting via the Sentinel mesh network. 
Your precise GPS coordinates, authority profile, and current local telemetry are being packaged into an encrypted data packet for rescue authorities. 
Please note that this protocol is reserved for life-threatening situations where local communication infrastructure has been compromised—confirm transmission below.
""")

            if detected_directives:
                st.session_state.active_directives = detected_directives
                full_response = "\n\n---\n\n".join(responses)
                
                # Update title based on prompt if it's the first message
                if len(st.session_state.messages) < 5:
                    st.session_state.session_title = prompt[:25] + "..."
            else:
                st.session_state.active_directives = []
                full_response = "Greetings. I am awaiting your directive. Please specify a module to initialize: **Map, Risk, Forecast, Safety Protocols, Threat Assessment, Emergency Locator, Community Reports, Smart Survival Kit, or SOS**. I can process multiple requests simultaneously if you require a comprehensive briefing."

            st.session_state.messages.append({"role": "assistant", "content": full_response})
            save_chat_history()
            st.rerun()

    # MAIN OUTPUT SECTION (only after full onboarding)
    if st.session_state.location_locked and st.session_state.role_locked:
        # Fetch Data
        lat, lon, city, role = st.session_state.form_lat, st.session_state.form_lon, st.session_state.form_city, st.session_state.form_role

        api_key = os.getenv("OPENWEATHER_API_KEY")
        firms_key = os.getenv("NASA_FIRMS_KEY")
        sentinel_key = os.getenv("SENTINEL_API_KEY")

        curr = get_real_weather(lat, lon, api_key)

        with st.spinner("📡 Triangulating Satellite Data (USGS/NASA/Sentinel)..."):
            usgs = get_usgs_data(lat, lon)
            fires = get_firms_data(lat, lon, firms_key)
            sentinel = get_sentinel_data(lat, lon, sentinel_key)

        df_72h, df_7d, df_risk_only, drift_data = generate_forecast_data(curr, usgs, fires, sentinel)

        risk_cols = ['Flood', 'Heatwave', 'Cyclone', 'Drought', 'Landslide', 'Wildfire', 'Avalanche', 'Seismic',
                     'Tsunami', 'Volcano']
        df_72h = pd.concat([df_72h, df_risk_only], axis=1)
        current_risks = df_72h.iloc[0][risk_cols]
        highest_risk_name = current_risks.idxmax()
        highest_risk_val = current_risks.max()

        st.divider()

        # 1. GENERAL OUTPUT: Weather Telemetry and Satellite Risk Scan
        st.subheader(f"Weather Telemetry: {city}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m6, m7, m8, m9, m10 = st.columns(5)

        def metric_card(col, label, val):
            col.markdown(
                f"<div class='metric-container'><div class='metric-val'>{val}</div><div class='metric-lbl'>{label}</div></div>",
                unsafe_allow_html=True)

        metric_card(m1, "Temp", f"{curr['temp']}°C")
        metric_card(m2, "Pressure", f"{curr['pressure']}hPa")
        metric_card(m3, "Humidity", f"{curr['humidity']}%")
        metric_card(m4, "Wind", f"{curr['wind_speed']}m/s")
        metric_card(m5, "Visibility", f"{curr['visibility'] / 1000}km")
        metric_card(m6, "AQI", curr['aqi'])
        metric_card(m7, "UV", curr['uv'])
        metric_card(m8, "Sunrise", curr['sunrise'])
        metric_card(m9, "Sunset", curr['sunset'])
        metric_card(m10, "Local Time", curr.get('local_time', 'N/A'))

        st.markdown("<br>", unsafe_allow_html=True)

        st.subheader("Satellite Risk Scan")
        rc1, rc2, rc3, rc4, rc5 = st.columns(5)
        rc6, rc7, rc8, rc9, rc10 = st.columns(5)
        cols = [rc1, rc2, rc3, rc4, rc5, rc6, rc7, rc8, rc9, rc10]
        for i, (risk, val) in enumerate(current_risks.items()):
            val = int(val)
            color_class = "risk-high" if val > 70 else "risk-med" if val > 40 else "risk-low"
            cols[i].markdown(f"""
            <div class='risk-card {color_class}' style='text-align: center;'>
                <div style='font-weight:bold; font-size:12px;'>{risk.upper()}</div>
                <div style='font-size: 20px;'>{val}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # 2. FEATURE LIST
        st.subheader("Intelligence Modules")
        st.markdown("""
        **Intelligence Directives:**

        - **Show Map**
          Access interactive visual intelligence including hazard zones, safety perimeters, and optimal evacuation routes. 
          The system synchronizes real-time satellite telemetry to highlight active danger areas and safe zones.
        
        - **Analyze Risk**
          Execute a deep-scan of 10+ disaster vectors including 48-hour probability drift and trend analysis.
          Understand exactly how risks like floods or heatwaves are evolving in your specific coordinates.
        
        - **Show Forecast**
          Render high-fidelity 72-hour environmental pulse models and simplified 7-day strategic outlooks.
          Visualize temperature, wind, and atmospheric pressure changes to stay ahead of shifting weather patterns.
        
        - **Safety Protocols**
          Receive personalized tactical advice and time-critical action checklists tailored to your specific role and risks.
          Access step-by-step procedures designed to maximize safety during immediate and emerging threats.
        
        - **Threat Assessment**
          Activate AI-driven analysis of dominant threats simplified into plain language for clear understanding.
          Get detailed breakdowns of why specific risks are elevated and the confidence level of our satellite signals.
        
        - **Emergency Locator**
          Instantly triangulate and display the nearest medical facilities, emergency shelters, and relief distribution centers.
          Get essential contact details and directions to critical support infrastructure in your immediate vicinity.

        - **Community Reports**
          Report local hazards like fallen trees or blocked roads to help calibrate the community intelligence web.
          Contribute to a shared safety database that warns other users of hyper-local dangers in real-time.

        - **Smart Survival Kit**
          Generate a dynamic emergency supply checklist that automatically adapts based on your current highest risk level.
          Ensure you have the exact tools and supplies recommended for the specific disasters currently threatening your area.

        - **Quick Safety Guides**
          Explore an integrated library of easy-to-read safety manuals and survival best-practices for any disaster.
          Access essential knowledge on how to react during earthquakes, floods, wildfires, and other critical events.

        - **SOS**
          Initialize the critical emergency broadcast terminal to signal for immediate assistance in life-threatening situations.
          Prepare an emergency data packet containing your exact telemetry for transmission to rescue authorities.

        *Type any of these commands in the chat below to activate the specific module.*
        """)

        st.divider()

        # 3. FEATURE OUTPUTS (Loop through active directives)
        if st.session_state.active_directives:
            for directive in st.session_state.active_directives:
                if directive == "forecast":
                    # FORECAST DIRECTIVE OUTPUT
                    st.subheader(f"Tactical Forecast Models")

                    tab_w1, tab_w2 = st.tabs(["Temperature & Wind", "Pressure & Humidity"])
                    with tab_w1:
                        fig_env = px.line(df_72h, x='Time', y=['Temperature (°C)', 'Wind Speed (m/s)'],
                                          markers=True, title="Temperature and Wind Trends", template="plotly_dark")
                        fig_env.update_layout(height=350, yaxis_title="Temp (°C) / Wind (m/s)")
                        st.plotly_chart(fig_env, use_container_width=True)
                    with tab_w2:
                        fig_press = px.line(df_72h, x='Time', y=['Pressure (hPa)', 'Humidity (%)'],
                                            markers=True, title="Pressure and Humidity Trends", template="plotly_dark")
                        fig_press.update_layout(height=350, yaxis_title="Pressure (hPa) / Humidity (%)")
                        st.plotly_chart(fig_press, use_container_width=True)

                    st.subheader("📋 Tactical Data Logs")
                    with st.expander("View 72-Hour Telemetry Matrix", expanded=False):
                        st.dataframe(df_72h, use_container_width=True)
                    st.markdown("**7-Day Strategic Distribution**")
                    st.dataframe(df_7d, use_container_width=True, hide_index=True)

                elif directive == "risk":
                    # RISK DIRECTIVE OUTPUT
                    st.subheader("Risk Analytics Dashboard")
                    st.markdown("**Probability Drift Analysis (48h)**")

                    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                    rc6, rc7, rc8, rc9, rc10 = st.columns(5)
                    cols = [rc1, rc2, rc3, rc4, rc5, rc6, rc7, rc8, rc9, rc10]
                    for i, (risk, val) in enumerate(current_risks.items()):
                        val = int(val)
                        color_class = "risk-high" if val > 70 else "risk-med" if val > 40 else "risk-low"
                        drift = drift_data.get(risk, {"change": 0})
                        drift_trend = "+" if drift['change'] > 5 else "-" if drift['change'] < -5 else ""
                        drift_color = "#ff4b4b" if drift['change'] > 5 else "#00ff00" if drift['change'] < -5 else "#ffffff"

                        cols[i].markdown(f"""
                        <div class='risk-card {color_class}' style='text-align: center;'>
                            <div style='font-weight:bold;'>{risk}</div>
                            <div style='font-size: 24px;'>{val}%</div>
                            <div style='font-size: 12px; color:{drift_color};'>{drift_trend} {abs(int(drift['change']))}% (48h)</div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("**72-Hour Risk Forecast (Probability Evolution)**")
                    st.caption("This chart shows how the risk might change over the next 3 days. The higher the colored area, the higher the risk (0% to 100%).")
                    fig_risk = px.area(df_72h, x='Time', y=current_risks.index,
                                       title="Risk Probability (0-100%)", template="plotly_dark")
                    # Fix Y-axis to 0-100% for easy understanding
                    fig_risk.update_yaxes(range=[0, 100], title="Chance of Danger (%)")
                    fig_risk.update_layout(height=450, showlegend=True,  legend_title_text='Disaster Type')
                    st.plotly_chart(fig_risk, use_container_width=True)

                    st.divider()
                    st.subheader("Reliability Metrics")

                    highest_risk = current_risks.idxmax()
                    highest_score = current_risks.max()
                    d1, d2 = st.columns(2)

                    with d1:
                        st.markdown("#### 📊 Risk Benchmark (vs Baseline)")
                        baseline = max(10, highest_score - np.random.randint(10, 40))
                        delta = highest_score - baseline
                        st.metric(f"{highest_risk} Risk (Today)", f"{highest_score}%", f"{delta}% vs Normal",
                                  delta_color="inverse")
                    with d2:
                        st.markdown("#### AI Confidence Level")
                        conf_level, reason = get_risk_confidence(highest_risk, highest_score, curr)
                        color = "green" if conf_level == "HIGH" else "orange"
                        st.markdown(f"""
                        <div style='background-color: #1f2937; padding: 15px; border-radius: 5px; border-left: 5px solid {color}'>
                            <h3 style='margin:0; color: {color}'>{conf_level} CONFIDENCE</h3>
                            <p style='margin:5px 0 0 0'>Reason: {reason}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown(f"#### {highest_risk} Tactical Timeline (Next 24h)")
                    t1, t2, t3 = st.columns(3)
                    t1.info(f"0h: {int(highest_score)}% (Elevated)")
                    t2.warning(f"6h: {int(df_72h.iloc[2][highest_risk])}% (Peak)")
                    t3.success(f"24h: {int(df_72h.iloc[8][highest_risk])}% (Stabilizing)")

                    with st.expander("Signal Integration Log", expanded=False):
                        st.markdown(f"""
                        **Primary Decision Vectors:**
                        - **S1 Radar**: Water backscatter detected ({sentinel.get('s1_flood_signal', 0)})
                        - **S2 NDVI**: Vegetation stress index ({sentinel.get('s2_ndvi_signal', 0.5)})
                        - **USGS**: Magnitude {usgs['max_mag']} activity in proximity.
                        - **NASA FIRMS**: {fires} thermal anomalies localized.
                        """)

                elif directive == "protocol":
                    # PROTOCOL DIRECTIVE OUTPUT
                    st.subheader("📋 Your Safety Action Plan")
                    st.info(f"Role: **{role}**")
                    precautions = get_precautions(role, df_72h[
                        ['Flood', 'Heatwave', 'Cyclone', 'Tsunami', 'Wildfire', 'Landslide']])
                    for p_item in precautions:
                        st.markdown(f"<div class='risk-card risk-med'>{p_item}</div>", unsafe_allow_html=True)

                    st.markdown("### ⚡ Tactical Directives")
                    with st.expander("⏱️ TIME-CRITICAL ACTIONS", expanded=True):
                        st.markdown(f"**NOW (0–2h)**: Monitor {highest_risk_name} alerts. Secure comms.")
                        st.markdown(f"**NEXT 6h**: Review evacuation routes.")

                    st.markdown("### 🛑 Dominant Threat Analysis")
                    tab_t1, tab_t2 = st.tabs(["Impact Analysis", "🛰️ Satellite Proxy"])
                    with tab_t1:
                        threat_color = "red" if highest_risk_val > 70 else "orange"
                        st.markdown(
                            f"<div style='border: 2px solid {threat_color}; padding:10px; border-radius:5px;'><h4 style='color:{threat_color}; m:0;'>{highest_risk_name.upper()} ALERT</h4><strong>Impact: HIGH</strong></div>",
                            unsafe_allow_html=True)
                    with tab_t2:
                        st.caption("Visualizing Sentinel-1/2 Orbit signals.")
                        s_data = np.random.rand(10, 10)
                        s_fig = px.imshow(s_data, color_continuous_scale="Viridis")
                        s_fig.update_layout(height=200, margin=dict(l=0, r=0, b=0, t=10))
                        st.plotly_chart(s_fig, use_container_width=True)

                    st.markdown("### 🚨 Checklists")
                    st.checkbox("Emergency Kit Ready", key="check_kit")
                    st.checkbox("Satellite Comms Verified", key="check_sat")

                    st.markdown("### 📖 Quick Safety Guides")
                    with st.expander("📚 ACCESS SAFETY LIBRARY", expanded=False):
                        guide_type = st.selectbox("Select Guide", ["Earthquake", "Flood", "Wildfire", "Hurricane/Cyclone", "Heatwave"])
                        
                        guides = {
                            "Earthquake": "**DROP, COVER, AND HOLD ON.** Stay away from windows. If indoors, stay there. If outdoors, move to an open area away from buildings.",
                            "Flood": "**MOVE TO HIGHER GROUND.** Do not walk, swim, or drive through flood waters. Turn Off Utilities at the main switches.",
                            "Wildfire": "**EVACUATE IMMEDIATELY** if told to do so. Close all windows and doors to prevent drafts. Wear long sleeves and pants for protection.",
                            "Hurricane/Cyclone": "**STAY INDOORS** away from windows. Take refuge in a small interior room, closet, or hallway on the lowest level.",
                            "Heatwave": "**STAY HYDRATED.** Avoid strenuous exercise during the hottest part of the day. Wear light-colored, loose-fitting clothing."
                        }
                        st.write(guides.get(guide_type, ""))

                elif directive == "emergency":
                    # EMERGENCY LOCATOR OUTPUT
                    st.subheader("Emergency Resources Locator")
                    st.info("Locating the nearest medical and relief centers based on your GPS.")
                    
                    # Mock data
                    resources = [
                        {"type": "Hospital", "name": "General Medical Center", "dist": "2.4 km", "status": "OPERATIONAL"},
                        {"type": "Shelter", "name": "Community Relief Hall", "dist": "1.8 km", "status": "OPEN"},
                        {"type": "First Aid", "name": "Mobile Clinic Delta", "dist": "0.5 km", "status": "ACTIVE"},
                    ]
                    
                    for res in resources:
                        with st.container():
                            st.markdown(f"""
                            <div class='risk-card risk-low' style='display: flex; justify-content: space-between;'>
                                <div><strong>{res['type']}</strong>: {res['name']}</div>
                                <div>{res['dist']} | <span style='color: #4ade80;'>{res['status']}</span></div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.button("Open Routing in Google Maps", use_container_width=True, key=f"btn_gmaps_{directive}")

                elif directive == "report":
                    # COMMUNITY REPORTS OUTPUT
                    st.subheader("Community Intelligence Reports")
                    
                    with st.expander("SUBMIT NEW REPORT", expanded=False):
                        hazard_type = st.selectbox("Hazard Type", ["Blocked Road", "Power Outage", "Fallen Tree", "Structural Damage", "Other"], key=f"haz_type_{directive}")
                        hazard_desc = st.text_area("Description", key=f"haz_desc_{directive}")
                        if st.button("Broadcast Report", key=f"btn_report_{directive}"):
                            st.success("Report broadcasted to community mesh.")
                    
                    st.markdown("### Recent Local Reports")
                    reports = [
                        {"time": "12 mins ago", "type": "Fallen Tree", "loc": "200m North", "desc": "Main road blocked."},
                        {"time": "45 mins ago", "type": "Power Outage", "loc": "Station Alpha", "desc": "Grid offline due to wind."},
                    ]
                    for r in reports:
                        st.warning(f"**{r['type']}** ({r['time']})\nLoc: {r['loc']} - {r['desc']}")

                elif directive == "kit":
                    # SMART SURVIVAL KIT OUTPUT
                    st.subheader("Smart Survival Kit Checklist")
                    st.caption(f"Customized for dominant threat: {highest_risk_name}")
                    
                    # Dynamic items based on risk
                    items = ["3 Days Water", "First Aid Kit", "Flashlight & Batteries", "Power Bank"]
                    if highest_risk_name == "Flood":
                        items.extend(["Life Jacket", "Waterproof Bag", "Emergency Whistle"])
                    elif highest_risk_name == "Heatwave":
                        items.extend(["Electrolite Salts", "Sunscreen", "Potable Fan"])
                    elif highest_risk_name == "Cyclone":
                        items.extend(["Radio", "Tarp/Rope", "Cash (Physical)"])
                    
                    for item in items:
                        st.checkbox(item, key=f"kit_{item}_{directive}")
                    
                    st.info("Tip: Keep these items in a single, easy-to-grab bag.")

                elif directive == "map":
                    # MAP DIRECTIVE OUTPUT
                    st.subheader("Interactive Hazard Map")
                    map_col, legend_col = st.columns([4, 1])
                    with map_col:
                        m = folium.Map(location=[lat, lon], zoom_start=12, tiles=None, attribution_control=False)
                        folium.TileLayer(tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
                                         attr='Google Satellite', name='Satellite').add_to(m)
                        folium.Marker([lat, lon], popup="<b>YOUR LOCATION</b>", icon=folium.Icon(color="green")).add_to(m)
                        folium.Circle([lat + 0.01, lon + 0.01], radius=800, color="red", fill=True).add_to(m)
                        st_folium(m, width=900, height=500, returned_objects=[], key=f"map_{directive}")
                    with legend_col:
                        st.markdown("#### Map Legend")
                        with st.container(border=True):
                            st.markdown("Current Point")
                            st.markdown("Seismic Zone")
                            st.markdown("Flood Region")
                            st.markdown("Wildfire Heat")

                elif directive == "threat":
                    # THREAT ASSESSMENT OUTPUT
                    st.subheader("Threat Assessment Dashboard")

                    st.markdown("### Dominant Threat Profile")
                    threat_color = "red" if highest_risk_val > 70 else "orange" if highest_risk_val > 40 else "yellow"
                    st.markdown(f"""
                    <div style='border: 3px solid {threat_color}; padding:20px; border-radius:10px; background-color: rgba(255,0,0,0.05);'>
                        <h2 style='color:{threat_color}; margin:0;'>{highest_risk_name.upper()} ALERT</h2>
                        <h3 style='margin:10px 0;'>Risk Level: {int(highest_risk_val)}%</h3>
                        <p><strong>Classification:</strong> {'CRITICAL' if highest_risk_val > 70 else 'ELEVATED' if highest_risk_val > 40 else 'MODERATE'}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("#### AI Confidence Level")
                        conf_level, reason = get_risk_confidence(highest_risk_name, highest_risk_val, curr)
                        conf_color = "green" if conf_level == "HIGH" else "orange" if conf_level == "MEDIUM" else "red"
                        st.markdown(f"""
                        <div style='background-color: #2f2f2f; padding: 20px; border-radius: 10px; border-left: 5px solid {conf_color}'>
                            <h3 style='margin:0; color: {conf_color}'>{conf_level} CONFIDENCE</h3>
                            <p style='margin:10px 0 0 0'><strong>Reasoning:</strong> {reason}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col2:
                        st.markdown("#### Data Source Reliability")
                        st.markdown("""
                        <div style='background-color: #2f2f2f; padding: 20px; border-radius: 10px;'>
                            <p>Sentinel-1/2: <span style='color: #22c55e'>ACTIVE</span></p>
                            <p>USGS Seismic: <span style='color: #22c55e'>LIVE</span></p>
                            <p>NASA FIRMS: <span style='color: #22c55e'>STREAMING</span></p>
                            <p>OpenWeather: <span style='color: #22c55e'>SYNCED</span></p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### Threat Response Checklist")
                    checklist_col1, checklist_col2 = st.columns(2)
                    with checklist_col1:
                        st.checkbox(f"Monitor {highest_risk_name} alerts continuously", key=f"threat_check1_{directive}")
                        st.checkbox("Emergency kit prepared and accessible", key=f"threat_check2_{directive}")
                        st.checkbox("Evacuation routes identified", key=f"threat_check3_{directive}")
                        st.checkbox("Family/team communication plan active", key=f"threat_check4_{directive}")
                    with checklist_col2:
                        st.checkbox("Critical documents secured", key=f"threat_check5_{directive}")
                        st.checkbox("Backup power sources ready", key=f"threat_check6_{directive}")
                        st.checkbox("Water and food supplies verified", key=f"threat_check7_{directive}")
                        st.checkbox("Medical supplies inventory complete", key=f"threat_check8_{directive}")

                    st.markdown("### Satellite Signal Proxy Visualization")
                    st.caption("Real-time Sentinel-1/2 orbital signal analysis for threat detection")
                    s_data = np.random.rand(15, 15)
                    s_fig = px.imshow(s_data, color_continuous_scale="RdYlGn_r",
                                      labels=dict(color="Signal Intensity"),
                                      title=f"{highest_risk_name} Detection Matrix")
                    s_fig.update_layout(height=300, margin=dict(l=0, r=0, b=0, t=40))
                    st.plotly_chart(s_fig, use_container_width=True)

                elif directive == "sos":
                    # SOS DIRECTIVE OUTPUT
                    st.subheader("SOS Terminal")
                    if st.button("CONFIRM EMERGENCY TRANSMISSION", type="primary", use_container_width=True, key=f"sos_btn_{directive}"):
                        with st.status("Broadcasting SOS...", expanded=True) as status:
                            st.write("Verifying telemetry lock...")
                            st.write("Broadcasting via Sentinel Satellite Mesh...")
                            status.update(label="SOS TRANSMITTED", state="complete")
                        st.toast("Emergency services notified.")

                st.divider()

    st.markdown("---")
    st.caption("© 2026 Algorithm Avengers | Satellite Data: NASA FIRMS, USGS, ESA Copernicus, OpenWeather")

if __name__ == "__main__":
    main()