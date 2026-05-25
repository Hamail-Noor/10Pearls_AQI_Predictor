import os
import datetime
import requests
import numpy as np
import pandas as pd

WEATHER_KEY = os.environ.get("OPENWEATHER_API_KEY")
CITY_TARGET = "Rawalpindi"
FEATURE_STORE_FILE = "vertex_feature_store_matrix.csv"

def run_hourly_extraction():
    print(f"🎬 Executing Hourly Feature Generation for {CITY_TARGET}...")
    
    # 1. Resolve geographic coordinates
    geo_url = f"http://api.openweathermap.org/geo/1.0/direct?q={CITY_TARGET}&limit=1&appid={WEATHER_KEY}"
    geo_res = requests.get(geo_url).json()
    if not geo_res:
        raise ValueError("Could not resolve city coordinates.")
    lat, lon = geo_res[0]['lat'], geo_res[0]['lon']
    
    # 2. Grab raw values from API
    aqi_url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={WEATHER_KEY}"
    aqi_res = requests.get(aqi_url).json()
    base_components = aqi_res['list'][0]['components']
    base_aqi = float(aqi_res['list'][0]['main']['aqi'])
    
    # 3. Handle data frames checking for existing records
    if os.path.exists(FEATURE_STORE_FILE):
        df = pd.read_csv(FEATURE_STORE_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        print("ℹ️ Feature Store file missing. Initializing standard baseline row...")
        df = pd.DataFrame()

    # 4. Engineer fresh row mapping features
    new_row = base_components.copy()
    current_time = datetime.datetime.utcnow()
    
    new_row['aqi'] = base_aqi
    new_row['timestamp'] = current_time
    new_row['city'] = CITY_TARGET
    new_row['hour'] = current_time.hour
    new_row['day'] = current_time.day
    new_row['month'] = current_time.month
    
    new_df = pd.DataFrame([new_row])
    df = pd.concat([df, new_df], ignore_index=True).sort_values(by='timestamp').reset_index(drop=True)
    
    # 5. Compute mandated derived change rate metric
    df['aqi_change_rate'] = df['aqi'].diff().fillna(0.0)
    
    # Commit changes back to storage
    df.to_csv(FEATURE_STORE_FILE, index=False)
    print(f"✅ Success. Feature Store updated. Total database scale: {len(df)} records.")

if __name__ == "__main__":
    if not WEATHER_KEY:
        raise ValueError("API token missing from Environment Variables.")
    run_hourly_extraction()