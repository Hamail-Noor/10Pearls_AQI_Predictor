import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def run_daily_training():
    print("📥 Loading features from local Feature Store array...")
    if not pd.io.common.file_exists("vertex_feature_store_matrix.csv"):
        print("❌ Error: Feature Store matrix file not found. Skipping training.")
        return
        
    df = pd.read_csv("vertex_feature_store_matrix.csv")
    
    # If the file hasn't accumulated a 'temp' column from the cloud pipeline yet, 
    # we inject a temporary placeholder so the script doesn't crash on your machine right now
    if 'temp' not in df.columns:
        df['temp'] = 25.0 + 5.0 * np.sin(df['hour'] / 24.0)

    # 1. Feature Extraction Layout
    tracking_features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'hour', 'day', 'month', 'aqi_change_rate']
    target_columns = ['aqi', 'temp'] # <-- Look at that! TWO target variables!

    X = df[tracking_features].values
    y = df[target_columns].values # <-- Multi-output array

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # ─── MODEL 1: MULTI-OUTPUT RIDGE REGRESSION ───
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train_scaled, y_train)
    preds_ridge = model_ridge.predict(X_test_scaled)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, preds_ridge))

    # ─── MODEL 2: MULTI-OUTPUT RANDOM FOREST ───
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    preds_rf = model_rf.predict(X_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(y_test, preds_rf))

    # ─── MODEL 3: MULTI-OUTPUT TENSORFLOW DNN ───
    model_tf = models.Sequential([
        layers.Input(shape=(12,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(2) # <-- Changed from 1 to 2! The network now predicts two numbers simultaneously!
    ])
    model_tf.compile(optimizer='adam', loss='mse')
    model_tf.fit(X_train_scaled, y_train, epochs=40, batch_size=16, verbose=0)
    preds_tf = model_tf.predict(X_test_scaled, verbose=0)
    rmse_tf = np.sqrt(mean_squared_error(y_test, preds_tf))

    print("\n" + "="*50 + "\n🔥 NEW MULTI-OUTPUT LEADERBOARD SCORECARD 🔥\n" + "="*50)
    print(f"• Ridge Baseline Combined RMSE : {rmse_ridge:.4f}")
    print(f"• Random Forest Combined RMSE   : {rmse_rf:.4f}")
    print(f"• TensorFlow DNN Combined RMSE  : {rmse_tf:.4f}")
    
    # Save the architecture to the registry vault
    model_tf.save("rawalpindi_aqi_model.h5")
    print("📁 Multi-Output Model Saved: 'rawalpindi_aqi_model.h5'")
    print("="*50)

if __name__ == "__main__":
    run_daily_training()