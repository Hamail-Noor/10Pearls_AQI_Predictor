import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def run_daily_training():
    print("📥 Loading features from local Feature Store array...")
    if not pd.io.common.file_exists("vertex_feature_store_matrix.csv"):
        print("❌ Error: Feature Store matrix file not found. Skipping training.")
        return
        
    df = pd.read_csv("vertex_feature_store_matrix.csv")
    if len(df) < 10:
        print("ℹ️ Dataset size too small for structural split validation. Accumulating more rows.")
        return

    tracking_features = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3', 'hour', 'day', 'month', 'aqi_change_rate']
    X = df[tracking_features].values
    y = df['aqi'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")

    # ─── MODEL 1: RIDGE REGRESSION ───
    model_ridge = Ridge(alpha=1.0)
    model_ridge.fit(X_train_scaled, y_train)
    preds_ridge = model_ridge.predict(X_test_scaled)
    rmse_ridge = np.sqrt(mean_squared_error(y_test, preds_ridge))

    # ─── MODEL 2: RANDOM FOREST ───
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X_train_scaled, y_train)
    preds_rf = model_rf.predict(X_test_scaled)
    rmse_rf = np.sqrt(mean_squared_error(y_test, preds_rf))

    # ─── MODEL 3: TENSORFLOW DNN ───
    model_tf = models.Sequential([
        layers.Input(shape=(12,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model_tf.compile(optimizer='adam', loss='mse')
    model_tf.fit(X_train_scaled, y_train, epochs=30, batch_size=16, verbose=0)
    preds_tf = model_tf.predict(X_test_scaled, verbose=0).flatten()
    rmse_tf = np.sqrt(mean_squared_error(y_test, preds_tf))

    print("\n" + "="*50 + "\nDAILY LEADERBOARD SCORECARD\n" + "="*50)
    print(f"• Ridge Regression RMSE : {rmse_ridge:.4f}")
    print(f"• Random Forest RMSE    : {rmse_rf:.4f}")
    print(f"• TensorFlow DNN RMSE   : {rmse_tf:.4f}")
    
    # Save the deep learning layout as a verified fallback deployment artifact
    model_tf.save("rawalpindi_aqi_model.h5")
    print("📁 Model Registry Vault Updated: 'rawalpindi_aqi_model.h5' saved.")
    print("="*50)

if __name__ == "__main__":
    run_daily_training()